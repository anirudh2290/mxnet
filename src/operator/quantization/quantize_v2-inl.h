/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize_v2-inl.h
 * \brief implementation of quantize operation
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_QUANTIZE_V2_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_QUANTIZE_V2_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <limits>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "./quantization_utils.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct QuantizeV2Param : public dmlc::Parameter<QuantizeV2Param> {
  enum OutType { kAuto = 0, kInt8, kUint8 };
  int out_type;
  dmlc::optional<float> min_calib_range;
  dmlc::optional<float> max_calib_range;
  int in_type;
  DMLC_DECLARE_PARAMETER(QuantizeV2Param) {
    DMLC_DECLARE_FIELD(out_type)
      .add_enum("auto", kAuto)
      .add_enum("int8", kInt8)
      .add_enum("uint8", kUint8)
      .set_default(kInt8)
      .describe("Output data type. `auto` can be specified to automatically determine output type "
                "according to min_calib_range.");
    DMLC_DECLARE_FIELD(min_calib_range)
      .set_default(dmlc::optional<float>())
      .describe("The minimum scalar value in the form of float32. If present, it will be used to "
                "quantize the fp32 data into int8 or uint8.");
    DMLC_DECLARE_FIELD(max_calib_range)
      .set_default(dmlc::optional<float>())
      .describe("The maximum scalar value in the form of float32. If present, it will be used to "
                "quantize the fp32 data into int8 or uint8.");
    DMLC_DECLARE_FIELD(in_type)
    .describe("in type");
  }
};

static mshadow::TypeFlag GetOutputType(const QuantizeV2Param &param) {
  auto out_type = mshadow::kInt8;
  if (param.out_type == QuantizeV2Param::OutType::kAuto) {
    if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
      if (param.min_calib_range.value() >= 0.0) {
        out_type = mshadow::kUint8;
      } else {
        out_type = mshadow::kInt8;
      }
    }
  } else if (param.out_type == QuantizeV2Param::OutType::kInt8) {
    out_type = mshadow::kInt8;
  } else if (param.out_type == QuantizeV2Param::OutType::kUint8) {
    out_type = mshadow::kUint8;
  } else {
    LOG(FATAL) << "Unsupported out_type in params: " <<param.out_type;
  }
  return out_type;
}

// quantize float to uint8_t
struct quantize_v2_unsigned {
  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, float *omin_range, float *omax_range,
                                  const SrcDType *in, const float imin_range,
                                  const float imax_range, const double min_limit,
                                  const double max_limit) {
    const float scale = (max_limit - min_limit) / (imax_range - imin_range);
    out[i] = static_cast<DstDType>((in[i] - imin_range) * scale + 0.5);
    *omin_range = imin_range;
    *omax_range = imax_range;
  }

  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, float *omin_range, float *omax_range,
                                  const SrcDType *in, const float *imin_range,
                                  const float *imax_range, const double min_limit,
                                  const double max_limit) {
    Map(i, out, omin_range, omax_range, in, *imin_range, *imax_range, min_limit, max_limit);
  }
};

// keep zero-center
struct quantize_v2_zero_centered {
  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, float *omin_range, float *omax_range,
                                  const SrcDType *in, const float imin_range,
                                  const float imax_range, const float quantized_range) {
    float real_range = MaxAbs(imin_range, imax_range);
    float scale = quantized_range / real_range;
    SrcDType x = in[i];
    out[i] = static_cast<DstDType>(Sign(x) * Min(Abs(x) * scale + 0.5f, quantized_range));
    *omin_range = -real_range;
    *omax_range = real_range;
  }

  template <typename DstDType, typename SrcDType>
  MSHADOW_XINLINE static void Map(int i, DstDType *out, float *omin_range, float *omax_range,
                                  const SrcDType *in, const float *imin_range,
                                  const float *imax_range, const float quantized_range) {
    Map(i, out, omin_range, omax_range, in, *imin_range, *imax_range, quantized_range);
  }
};

template <typename xpu>
void QuantizeV2Compute(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                       const std::vector<TBlob> &inputs, const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  typedef float SrcDType;
  using mshadow::red::limits::MaxValue;
  using mshadow::red::limits::MinValue;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const QuantizeV2Param &param = nnvm::get<QuantizeV2Param>(attrs.parsed);
  auto out_type = GetOutputType(param);
  if (param.min_calib_range.has_value() && param.max_calib_range.has_value()) {
    if (out_type == mshadow::kUint8) {
      Kernel<quantize_v2_unsigned, xpu>::Launch(
          s, outputs[0].Size(), outputs[0].dptr<uint8_t>(), outputs[1].dptr<float>(),
          outputs[2].dptr<float>(), inputs[0].dptr<SrcDType>(), param.min_calib_range.value(),
          param.max_calib_range.value(), MinValue<uint8_t>(), MaxValue<uint8_t>());
    } else if (out_type == mshadow::kInt8) {  // zero-centered quantization
      Kernel<quantize_v2_zero_centered, xpu>::Launch(
          s, outputs[0].Size(), outputs[0].dptr<int8_t>(), outputs[1].dptr<float>(),
          outputs[2].dptr<float>(), inputs[0].dptr<SrcDType>(), param.min_calib_range.value(),
          param.max_calib_range.value(), MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()));
    } else {
      LOG(FATAL) << "quantize op only supports int8 and uint8 as output type";
    }
  } else {  // model is not calibrated
    TShape src_shape, dst_shape;
    const size_t actual_float_size = sizeof(float);
    const size_t temp_reduce_size =
        ConfigReduce<xpu, SrcDType>(s, inputs[0].shape_, TShape({1}), &src_shape, &dst_shape);
    Tensor<xpu, 1, char> temp_space = ctx.requested[0].get_space_typed<xpu, 1, char>(
        Shape1(2 * actual_float_size + temp_reduce_size), s);
    const int dev_id = ctx.run_ctx.ctx.dev_id;
    TBlob in_min_t(reinterpret_cast<SrcDType *>(temp_space.dptr_), Shape1(1), xpu::kDevMask,
                   dev_id);
    TBlob in_max_t(reinterpret_cast<SrcDType *>(temp_space.dptr_) + 1, Shape1(1), xpu::kDevMask,
                   dev_id);
    Tensor<xpu, 1, char> workspace(temp_space.dptr_ + 2 * actual_float_size,
                                   Shape1(temp_reduce_size), s);
    broadcast::Reduce<red::minimum, 2, SrcDType, mshadow::op::identity>(
        s, in_min_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
    broadcast::Reduce<red::maximum, 2, SrcDType, mshadow::op::identity>(
        s, in_max_t.reshape(dst_shape), kWriteTo, workspace, inputs[0].reshape(src_shape));
    if (out_type == mshadow::kUint8) {
      Kernel<quantize_v2_unsigned, xpu>::Launch(
          s, outputs[0].Size(), outputs[0].dptr<uint8_t>(), outputs[1].dptr<float>(),
          outputs[2].dptr<float>(), inputs[0].dptr<SrcDType>(), in_min_t.dptr<float>(),
          in_max_t.dptr<float>(), MinValue<uint8_t>(), MaxValue<uint8_t>());
    } else if (out_type == mshadow::kInt8) {  // zero-centered quantization
      Kernel<quantize_v2_zero_centered, xpu>::Launch(
          s, outputs[0].Size(), outputs[0].dptr<int8_t>(), outputs[1].dptr<float>(),
          outputs[2].dptr<float>(), inputs[0].dptr<SrcDType>(), in_min_t.dptr<float>(),
          in_max_t.dptr<float>(), MinAbs(MaxValue<int8_t>(), MinValue<int8_t>()));
    } else {
      LOG(FATAL) << "quantize op only supports int8 and uint8 as output type";
    }
  }
}

static inline bool QuantizeV2Shape(const nnvm::NodeAttrs &attrs, std::vector<TShape> *in_attrs,
                                   std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 3U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*out_attrs, 1, TShape{1});
  SHAPE_ASSIGN_CHECK(*out_attrs, 2, TShape{1});
  return !shape_is_none(out_attrs->at(0));
}

static inline bool QuantizeV2Type(const nnvm::NodeAttrs &attrs, std::vector<int> *in_attrs,
                                  std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 3U);
  const QuantizeV2Param &param = nnvm::get<QuantizeV2Param>(attrs.parsed);
  if ((*in_attrs)[0] == mshadow::kUint8) {
    TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kUint8);
    param.in_type = mshadow::kUint8;
  } else if ((*in_attrs)[0] == mshadow::kInt8) {
    TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kInt8);
    param.in_type = mshadow::kInt8;
  } else {
    param.in_type = mshadow::kFloat32;
    TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
  }
  auto out_type = GetOutputType(param);
  if (out_type == mshadow::kUint8) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kUint8);
  } else if (out_type == mshadow::kInt8) {
    TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  } else {
    LOG(FATAL) << "quantize op only supports int8 and uint8 as output type";
  }
  TYPE_ASSIGN_CHECK(*out_attrs, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_attrs, 2, mshadow::kFloat32);
  return (*in_attrs)[0] != -1;
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_QUANTIZE_V2_INL_H_
