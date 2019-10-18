#include <time.h>
#include <dmlc/logging.h>
#include <dmlc/thread_group.h>
#include <dmlc/omp.h>
#include <mxnet/c_api.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/timer.h>
#include <cstdio>
#include <thread>
#include <chrono>
#include <vector>
#include "mxnet-cpp/MxNetCpp.h"

/*
 * Ops pushed from different threads: one thread per op, wait_to_read in each thread, followed by waitall in main thread
 */

using namespace mxnet;

inline void DerefInputOutput(const std::vector<NDArray*>& inputs,
                             const std::vector<NDArray*>& outputs,
                             std::vector<NDArray>* p_inputs,
                             std::vector<NDArray>* p_outputs) {
  p_inputs->reserve(inputs.size());
  p_outputs->reserve(outputs.size());
  for (NDArray* i : inputs) p_inputs->emplace_back(*i);
  for (NDArray* i : outputs) p_outputs->emplace_back(*i);
}

inline void InvalidateOutputs(const std::vector<NDArray> &arrs,
                                     const std::vector<OpReqType> &reqs) {
  for (size_t i = 0; i < arrs.size(); i++) {
    if (reqs[i] == kWriteTo || reqs[i] == kNullOp) {
      const_cast<NDArray &>(arrs[i]).InvalidateMKLDNNData();
    }
  }
}

inline void SetDependency(const nnvm::NodeAttrs& attrs,
                   const mxnet::Context& ctx,
                   const std::vector<mxnet::NDArray*>& inputs,
                   const std::vector<mxnet::NDArray*>& outputs,
                   std::vector<engine::VarHandle> *p_read_vars,
                   std::vector<engine::VarHandle> *p_write_vars,
                   std::vector<mxnet::Resource> *p_requested,
                   std::vector<uint32_t> *p_mutate_idx,
                   const DispatchMode dispatch_mode) {
  static auto& fmutate = nnvm::Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  static auto& ftmp_resource = nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");
  static auto& ftmp_resource_ex = nnvm::Op::GetAttr<FResourceRequestEx>("FResourceRequestEx");

  std::vector<engine::VarHandle>& read_vars  = *p_read_vars;
  std::vector<engine::VarHandle>& write_vars = *p_write_vars;
  std::vector<Resource>& requested = *p_requested;
  std::vector<uint32_t>& mutate_idx = *p_mutate_idx;

  if (fmutate.count(attrs.op)) {
    mutate_idx = fmutate[attrs.op](attrs);
  }
  const bool rsc_req = (ftmp_resource.count(attrs.op) != 0);
  const bool rsc_ex_req = (ftmp_resource_ex.count(attrs.op) != 0);
  if (rsc_req || rsc_ex_req) {
    int ntmp = 0;
    auto resource_reqs = rsc_ex_req ? ftmp_resource_ex[attrs.op](attrs,
                                          static_cast<int>(ctx.dev_mask()), dispatch_mode)
                                    : ftmp_resource[attrs.op](attrs);
    for (const auto& req : resource_reqs) {
      switch (req.type) {
       case ResourceRequest::kTempSpace:
        ++ntmp;
       case ResourceRequest::kRandom:
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
        write_vars.push_back(requested.back().var);
        break;
       case ResourceRequest::kParallelRandom:
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
        write_vars.push_back(requested.back().var);
        break;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
       case ResourceRequest::kCuDNNDropoutDesc:
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
        write_vars.push_back(requested.back().var);
        break;
#endif  // MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
       default:
        LOG(FATAL) << "resource type not yet supported";
      }
    }
    CHECK_LE(ntmp, 1) << "Only support 1 temp space request";
  }

  // append extra resource requests for storage fallback
  if (dispatch_mode == DispatchMode::kFComputeFallback) {
    requested.push_back(ResourceManager::Get()->Request(ctx, ResourceRequest::kTempSpace));
    write_vars.push_back(requested.back().var);
  }

  read_vars.reserve(inputs.size());
  for (auto& i : inputs) {
    read_vars.push_back(i->var());
  }
  write_vars.reserve(outputs.size() + mutate_idx.size());
  for (auto& i : outputs) {
    write_vars.push_back(i->var());
  }
  for (auto & i : mutate_idx) {
    write_vars.push_back(inputs[i]->var());
  }
  Engine::Get()->DeduplicateVarHandle(&read_vars, &write_vars);
}

template<typename FCompType>
FCompType GetFCompute(const nnvm::Op* op, const std::string& name,
                      const Context& ctx) {
  static auto& fcompute_cpu = nnvm::Op::GetAttr<FCompType>(name + "<cpu>");
  static auto& fcompute_gpu = nnvm::Op::GetAttr<FCompType>(name + "<gpu>");

  if (ctx.dev_mask() == cpu::kDevMask) {
    return fcompute_cpu.get(op, nullptr);
  } else if (ctx.dev_mask() == gpu::kDevMask) {
    return fcompute_gpu.get(op, nullptr);
  } else {
    LOG(FATAL) << "Unknown device mask " << ctx.dev_mask();
    return nullptr;
  }
}



int main(int argc, char const *argv[]) {
  /*Input data preparation*/
  CHECK(argc == 2) << "One argument expected : num_threads";
  int num_threads = atoi(argv[1]);
  //std::string input_ctx = std::string(argv[2]);
  std::string input_ctx = "cpu";
  //CHECK(input_ctx == "cpu") << "Bad input, only cpu context supported currently";
  mxnet::cpp::Context ctx = mxnet::cpp::Context::cpu();
  Context backend_ctx;
  if (input_ctx == "cpu") {
      ctx = mxnet::cpp::Context::cpu();
      backend_ctx = Context::CPU(0);
  } else {
      ctx = mxnet::cpp::Context::gpu(0);
      backend_ctx = Context::GPU(0);
  }
  const nnvm::Op *op = Op::Get("Convolution");
  nnvm::NodeAttrs attrs;
  attrs.op = op;
  attrs.name = "conv_node1";
  std::unordered_map<std::string, std::string> u = {{"kernel", "(2,2)"},
                {"no_bias", "0"},
                {"dilate", "(1,1)"},
                {"num_group", "1"},
                {"layout", "NCHW"},
                {"stride", "(1,1)"},
                {"pad", "(0,0)"},
                {"num_filter", "10"}};

  attrs.dict = u;
  op->attr_parser(&attrs);

  std::vector<mxnet::cpp::NDArray> data_arr, data_arr2;
  std::vector<mxnet::cpp::NDArray> weight_arr, weight_arr2;
  std::vector<mxnet::cpp::NDArray> bias_arr, bias_arr2;
  std::vector<mxnet::cpp::NDArray> output_arr, output_arr2;
  for (size_t i = 0; i < num_threads; ++i) {
    data_arr.emplace_back(mxnet::cpp::Shape(2, 4, 10, 10), ctx, false, 0);
    weight_arr.emplace_back(mxnet::cpp::Shape(10, 4, 2, 2), ctx, false, 0);
    bias_arr.emplace_back(mxnet::cpp::Shape(10), ctx, false, 0);
    output_arr.emplace_back(mxnet::cpp::Shape(2, 10, 9, 9), ctx, false, 0);
    data_arr2.emplace_back(mxnet::cpp::Shape(2, 4, 10, 10), ctx, false, 0);
    weight_arr2.emplace_back(mxnet::cpp::Shape(10, 4, 2, 2), ctx, false, 0);
    bias_arr2.emplace_back(mxnet::cpp::Shape(10), ctx, false, 0);
    output_arr2.emplace_back(mxnet::cpp::Shape(2, 10, 9, 9), ctx, false, 0);
    int begin = 1000 * i;
    int end = begin + 1000;
    mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(data_arr[i]);
    mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(weight_arr[i]);
    mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(bias_arr[i]);
    mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(data_arr2[i]);
    mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(weight_arr2[i]);
    mxnet::cpp::Operator("_random_uniform")(begin, end).Invoke(bias_arr2[i]);
    mxnet::cpp::NDArray::WaitAll();
  }

  /*Obtain expected results*/
    mxnet::cpp::Symbol data = mxnet::cpp::Symbol::Variable("data");
    mxnet::cpp::Symbol weight = mxnet::cpp::Symbol::Variable("weight");
    mxnet::cpp::Symbol bias = mxnet::cpp::Symbol::Variable("bias");
    auto out = mxnet::cpp::Operator("Convolution")
            .SetParam("kernel", mxnet::cpp::Shape(2, 2))
            .SetParam("no_bias", false)
            .SetParam("dilate", mxnet::cpp::Shape(1, 1))
            .SetParam("num_group", 1)
            .SetParam("layout", "NCHW")
            .SetParam("stride", mxnet::cpp::Shape(1, 1))
            .SetParam("pad", mxnet::cpp::Shape(0, 0))
            .SetParam("num_filter", 10)
            .SetInput("data", data)
            .SetInput("weight", weight)
            .SetInput("bias", bias)
            .CreateSymbol("fwd");
    CachedOpHandle hdl = CachedOpHandle();
    std::vector<std::string> flag_keys = {"data_indices", "param_indices"};
    std::vector<std::string> flag_vals {"[0]", "[1, 2]"};
    std::vector<const char*> flag_key_cstrs;
    flag_key_cstrs.reserve(flag_keys.size());
    for (size_t i = 0; i < flag_keys.size(); ++i) {
        flag_key_cstrs.emplace_back(flag_keys[i].c_str());
    }
    std::vector<const char*> flag_val_cstrs;
    flag_val_cstrs.reserve(flag_vals.size());
    for (size_t i = 0; i < flag_vals.size(); ++i) {
        flag_val_cstrs.emplace_back(flag_vals[i].c_str());
    }

    int ret1 = MXCreateCachedOpEx(out.GetHandle(),
                                  flag_keys.size(),
                                  flag_key_cstrs.data(),
                                  flag_val_cstrs.data(),
                                  &hdl, false);
    if (ret1 < 0) {
       LOG(INFO) << MXGetLastError();
    }

    std::vector<NDArrayHandle> arr_handles(3);
    std::vector<NDArrayHandle*> nd_ptrs(num_threads);
    std::vector<mxnet::cpp::NDArray> result_expected(num_threads);
    std::vector<mxnet::cpp::NDArray> result_expected2(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
    int num_output;

    arr_handles[0] = data_arr[i].GetHandle();
    arr_handles[1] = weight_arr[i].GetHandle();
    arr_handles[2] = bias_arr[i].GetHandle();
    const int* stypes;

    int ret4 = MXInvokeCachedOpEx(hdl, 3, arr_handles.data(), &num_output,
                                  &nd_ptrs[i], &stypes);
    if (ret4 < 0) {
        LOG(INFO) << MXGetLastError();
    }
    mxnet::cpp::NDArray::WaitAll();
    result_expected[i] = mxnet::cpp::NDArray(*nd_ptrs[i]);
    }

    std::string name_expected =
        "/home/ubuntu/experimentals/mxnet_thread_safe_poc/result_expected.params";
    mxnet::cpp::NDArray::Save(name_expected, result_expected);
    std::vector<NDArrayHandle> arr_handles2(3);
    std::vector<NDArrayHandle*> nd_ptrs2(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
    int num_output;

    arr_handles2[0] = data_arr2[i].GetHandle();
    arr_handles2[1] = weight_arr2[i].GetHandle();
    arr_handles2[2] = bias_arr2[i].GetHandle();
    const int* stypes;

    int ret4 = MXInvokeCachedOpEx(hdl, 3, arr_handles2.data(), &num_output,
                                  &nd_ptrs2[i], &stypes);
    if (ret4 < 0) {
        LOG(INFO) << MXGetLastError();
    }
    mxnet::cpp::NDArray::WaitAll();
    result_expected2[i] = mxnet::cpp::NDArray(*nd_ptrs2[i]);
    }
    std::string name_expected2 =
        "/home/ubuntu/experimentals/mxnet_thread_safe_poc/result_expected2.params";
    mxnet::cpp::NDArray::Save(name_expected2, result_expected2);



  /*Run multithreaded*/
  std::vector<mxnet::NDArray*> data_mx_arr, weight_mx_arr, bias_mx_arr, output_mx_arr;
  std::vector<mxnet::NDArray*> data_mx_arr2, weight_mx_arr2, bias_mx_arr2, output_mx_arr2;
  data_mx_arr.resize(num_threads);
  weight_mx_arr.resize(num_threads);
  bias_mx_arr.resize(num_threads);
  output_mx_arr.resize(num_threads);
  data_mx_arr2.resize(num_threads);
  weight_mx_arr2.resize(num_threads);
  bias_mx_arr2.resize(num_threads);
  output_mx_arr2.resize(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    data_mx_arr[i] = (NDArray*)data_arr[i].GetHandle();
    weight_mx_arr[i] = (NDArray*)weight_arr[i].GetHandle();
    bias_mx_arr[i] = (NDArray*)bias_arr[i].GetHandle();
    output_mx_arr[i] = (NDArray*)output_arr[i].GetHandle();
    data_mx_arr2[i] = (NDArray*)data_arr2[i].GetHandle();
    weight_mx_arr2[i] = (NDArray*)weight_arr2[i].GetHandle();
    bias_mx_arr2[i] = (NDArray*)bias_arr2[i].GetHandle();
    output_mx_arr2[i] = (NDArray*)output_arr2[i].GetHandle();
  }

  std::vector<NDArray*> inputs, outputs;

  for (size_t i = 0; i < num_threads; ++i) {
    inputs.emplace_back(data_mx_arr[i]);
    inputs.emplace_back(weight_mx_arr[i]);
    inputs.emplace_back(bias_mx_arr[i]);
    outputs.emplace_back(output_mx_arr[i]);
  }
  std::vector<NDArray*> inputs2, outputs2;
  for (size_t i = 0; i < num_threads; ++i) {
    inputs2.emplace_back(data_mx_arr2[i]);
    inputs2.emplace_back(weight_mx_arr2[i]);
    inputs2.emplace_back(bias_mx_arr2[i]);
    outputs2.emplace_back(output_mx_arr2[i]);
  }


  std::vector<std::vector<engine::VarHandle>> read_vars;
  std::vector<std::vector<engine::VarHandle>> write_vars;
  read_vars.resize(num_threads);
  write_vars.resize(num_threads);
  std::vector<std::vector<engine::VarHandle>> read_vars2;
  std::vector<std::vector<engine::VarHandle>> write_vars2;
  read_vars2.resize(num_threads);
  write_vars2.resize(num_threads);
  auto func = [&](int num) {
  //std::vector<engine::VarHandle> read_vars, write_vars;
  std::vector<Resource> requested, requested2;
  std::vector<uint32_t> mutate_idx, mutate_idx2;
  DispatchMode dispatch_mode = DispatchMode::kFComputeEx;
  DispatchMode dispatch_mode2 = DispatchMode::kFComputeEx;
  std::vector<NDArray*> tmp_inputs, tmp_outputs;
  std::vector<NDArray*> tmp_inputs2, tmp_outputs2;
  tmp_inputs.emplace_back(data_mx_arr[num]);
  tmp_inputs.emplace_back(weight_mx_arr[num]);
  tmp_inputs.emplace_back(bias_mx_arr[num]);
  tmp_outputs.emplace_back(output_mx_arr[num]);
  tmp_inputs2.emplace_back(data_mx_arr2[num]);
  tmp_inputs2.emplace_back(weight_mx_arr2[num]);
  tmp_inputs2.emplace_back(bias_mx_arr2[num]);
  tmp_outputs2.emplace_back(output_mx_arr2[num]);
  SetDependency(attrs, backend_ctx, tmp_inputs, tmp_outputs,
                &read_vars[num], &write_vars[num], &requested, &mutate_idx, dispatch_mode);
  SetDependency(attrs, backend_ctx, tmp_inputs2, tmp_outputs2,
                &read_vars2[num], &write_vars2[num], &requested2, &mutate_idx2, dispatch_mode2);
  std::vector<NDArray> p_inputs, p_outputs;

  DerefInputOutput(tmp_inputs, tmp_outputs, &p_inputs, &p_outputs);

  std::vector<NDArray> p_inputs2, p_outputs2;
  DerefInputOutput(tmp_inputs2, tmp_outputs2, &p_inputs2, &p_outputs2);
  FComputeEx fn_ex = GetFCompute<FComputeEx>(attrs.op, std::string("FComputeEx"), backend_ctx);

  bool is_train = false;
  bool need_grad = false;
  std::vector<OpReqType> reqs;
  reqs.push_back(kWriteTo);
  const auto& run = [=](RunContext rctx) {
      OpContext opctx{need_grad, is_train, rctx, engine::CallbackOnComplete(), requested};
      InvalidateOutputs(p_outputs, reqs);
      fn_ex(attrs, opctx, p_inputs, reqs, p_outputs);
      if (backend_ctx.dev_mask() == gpu::kDevMask && !rctx.is_bulk) {
          rctx.get_stream<gpu>()->Wait();
      }
    };

  const auto& run2 = [=](RunContext rctx) {
      OpContext opctx{need_grad, is_train, rctx, engine::CallbackOnComplete(), requested2};
      InvalidateOutputs(p_outputs2, reqs);
      fn_ex(attrs, opctx, p_inputs2, reqs, p_outputs2);
      if (backend_ctx.dev_mask() == gpu::kDevMask && !rctx.is_bulk) {
          rctx.get_stream<gpu>()->Wait();
      }
    };

    Engine::Get()->PushSync(run, backend_ctx, read_vars[num], write_vars[num], FnProperty::kNormal,
                            0, op->name.c_str());
    Engine::Get()->PushSync(run2, backend_ctx, read_vars2[num], write_vars2[num], FnProperty::kNormal,
                            0, op->name.c_str());
  };

  std::vector<std::thread> worker_threads(num_threads);
  int count = 0;
  for (auto&& i : worker_threads) {
      i = std::thread(func, count);
      count++;
  }
  for (auto&& i : worker_threads) {
      i.join();
  }
  mxnet::cpp::NDArray::WaitAll();
  for (size_t i = 0; i < num_threads; ++i) {
    for (const engine::VarHandle& var : write_vars[i]) {
        Engine::Get()->WaitForVar(var);
    }
  }

  for (size_t i = 0; i < num_threads; ++i) {
    for (const engine::VarHandle& var : write_vars2[i]) {
        Engine::Get()->WaitForVar(var);
    }
  }

  std::string name = "/home/ubuntu/experimentals/mxnet_thread_safe_poc/result.params";
  std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(name.c_str(), "w"));
  std::vector<std::string> op_names;
  std::vector<NDArray> final_outputs;
  final_outputs.resize(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    final_outputs[i] = *(outputs[i]);
  }
  mxnet::NDArray::Save(fo.get(), final_outputs, op_names);


  std::string name2 = "/home/ubuntu/experimentals/mxnet_thread_safe_poc/result2.params";
  std::unique_ptr<dmlc::Stream> fo2(dmlc::Stream::Create(name2.c_str(), "w"));
  std::vector<NDArray> final_outputs2;
  final_outputs2.resize(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    final_outputs2[i] = *(outputs2[i]);
  }
  mxnet::NDArray::Save(fo2.get(), final_outputs2, op_names);
}
