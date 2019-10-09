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

#include <unordered_set>
#include <iostream>
#include "./imperative_utils.h"
#include "../executor/exec_pass.h"
#include "./cached_op_threadsafe.h"
#include "../operator/operator_common.h"
#include "../operator/subgraph/common.h"

namespace mxnet {

DMLC_REGISTER_PARAMETER(CachedOpThreadSafeConfig);

constexpr uint32_t kEidNotExist = std::numeric_limits<uint32_t>::max();

struct CachedOpThreadSafe::CachedOpThreadSafeState {
  CachedOpThreadSafeState(const Context &context_,
                          const nnvm::Graph &fwd_graph_) {

    context = context_;
    info.fwd_graph = fwd_graph_;

    size_t max_nodes = info.fwd_graph.indexed_graph().num_nodes();
    size_t max_entries = info.fwd_graph.indexed_graph().num_node_entries();
    info.fwd_graph.attrs["context"] =
        std::make_shared<dmlc::any>(std::vector<Context>(
            info.fwd_graph.indexed_graph().num_nodes(), context));

    buff.resize(max_entries);
    arrays.resize(max_entries);
    array_reqs.resize(max_entries);
    dynamic_entries.resize(max_entries, false);
  }

  std::mutex mutex;
  Context context;
  mxnet::GraphInfo info;
  bool fwd_alloc = false;
  bool fwd_exec_init = false;

  std::vector<NDArray> buff;
  std::vector<NDArray*> arrays;
  std::vector<NDArray*> arrays_with_in_out;
  std::vector<OpReqType> array_reqs;

  std::vector<bool> dynamic_entries;
  std::multimap<size_t, NDArray> fwd_reuse_pool;

};



OpStatePtr CachedOpThreadSafe::GetCachedOpThreadSafeState(
    const Context& ctx) {

  for (const auto& i : cached_op_states_[ctx]) {
    // only create one state per device when not using static memory
    if (i.unique()) {
      return i;
    }
  }
  auto state_ptr = OpStatePtr::Create<CachedOpThreadSafeState>(ctx, fwd_graph_);

  cached_op_states_[ctx].push_back(state_ptr);
  return state_ptr;
}


CachedOpThreadSafe::CachedOpThreadSafe(const nnvm::Symbol& sym,
                                       const std::vector<std::pair<std::string, std::string> >& flags) {
  using namespace nnvm;
  using namespace imperative;
  static const std::vector<const Op*> zero_ops{Op::Get("zeros_like"), Op::Get("_zeros")};
  static const auto _copy_op = Op::Get("_copy");
  config_.Init(flags);

  // construct forward graph
  {
    NodeEntryMap<size_t> dedup_out;
    for (const NodeEntry &nodeEntry : sym.outputs) {
      if (dedup_out.find(nodeEntry) != dedup_out.end()) {
        NodePtr copy_node = Node::Create();
        copy_node->attrs.op = _copy_op;
        copy_node->attrs.name = nodeEntry.node->attrs.name + "_copy" +
                                std::to_string(dedup_out[nodeEntry]++);
        copy_node->inputs.emplace_back(nodeEntry);
        if (_copy_op->attr_parser != nullptr) {
          _copy_op->attr_parser(&(copy_node->attrs));
        }
        fwd_graph_.outputs.emplace_back(std::move(copy_node));
      } else {
        dedup_out.emplace(nodeEntry, 0);
        fwd_graph_.outputs.push_back(nodeEntry);
      }
    }

    const auto &idx = fwd_graph_.indexed_graph();
    CHECK_GE(idx.input_nodes().size(), 1)
        << "CachedOp requires at least 1 input";

    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    for (const auto &i : idx.input_nodes())
      ++ref_count[idx.entry_id(i, 0)];
    for (const auto &i : idx.outputs())
      ++ref_count[idx.entry_id(i)];
    for (size_t i = 0; i < idx.num_nodes(); ++i) {
      for (const auto &j : idx[i].inputs)
        ++ref_count[idx.entry_id(j)];
    }

    fwd_graph_.attrs["forward_ref_count"] =
        std::make_shared<dmlc::any>(std::move(ref_count));
  }

  // Set param indices
  {
    const auto& indexed_graph = fwd_graph_.indexed_graph();
    if (config_.data_indices.ndim() || config_.param_indices.ndim()) {
      CHECK_EQ(config_.data_indices.ndim() + config_.param_indices.ndim(),
               indexed_graph.input_nodes().size());
    } else {
      std::vector<uint32_t> tmp;
      tmp.reserve(indexed_graph.input_nodes().size());
      for (size_t i = 0; i < indexed_graph.input_nodes().size(); ++i) {
        tmp.emplace_back(i);
      }
      config_.data_indices.assign(tmp.begin(), tmp.end());
    }

  }

}

bool CachedOpThreadSafe::SetForwardGraph(mxnet::GraphInfo *info,
                                         const std::vector<NDArray *> &inputs) {
  using namespace nnvm;
  using namespace imperative;
  CHECK_EQ(inputs.size(), num_inputs());
  nnvm::Graph& g = info->fwd_graph;

  ShapeVector shape_inputs;
  DTypeVector dtype_inputs;
  StorageTypeVector storage_type_inputs;
  shape_inputs.reserve(inputs.size());
  dtype_inputs.reserve(inputs.size());
  storage_type_inputs.reserve(inputs.size());
  for (auto input : inputs) {
    shape_inputs.emplace_back(input->shape());
    dtype_inputs.emplace_back(input->dtype());
    storage_type_inputs.emplace_back(input->storage_type());
  }

  bool match = true;
  bool contain_dynamic_shape = false;
  match &= CheckAndInferShape(&g, std::move(shape_inputs), true,
                              {0, 0}, {0, 0}, &contain_dynamic_shape);
  match &= CheckAndInferType(&g, std::move(dtype_inputs), true);
  exec::DevMaskVector dev_mask(g.indexed_graph().num_nodes(), inputs[0]->ctx().dev_mask());
  match &= CheckAndInferStorageType(&g, std::move(dev_mask),
                                    std::move(storage_type_inputs), true);

  if (!match) {
    g.attrs.erase("forward_mem_plan");
    g.attrs.erase("full_mem_plan");
  } else if (g.attrs.count("forward_mem_plan")) {
    return true;
  }

  const auto& idx = g.indexed_graph();

  StorageVector storage(idx.num_node_entries(), exec::kBadStorageID);
  const auto& stypes = g.GetAttr<StorageTypeVector>("storage_type");
  CHECK_EQ(stypes.size(), storage.size());

  for (size_t i = 0; i < stypes.size(); i++) {
    if (stypes[i] != kDefaultStorage) storage[i] = exec::kDynamicStorageID;
  }

  for (const auto i : idx.input_nodes()) {
    storage[idx.entry_id(i, 0)] = exec::kExternalStorageID;
  }

  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    storage[idx.entry_id(idx.outputs()[i])] = exec::kExternalStorageID;
  }

  auto mem_plan = PlanMemory(
      &g, std::move(storage), g.GetAttr<std::vector<uint32_t> >("forward_ref_count"));
  g.attrs["forward_mem_plan"] =
      std::make_shared<dmlc::any>(std::move(mem_plan));

  return false;

}

OpStatePtr CachedOpThreadSafe::DynamicForward(const Context& default_ctx,
                                              const std::vector<NDArray*>& inputs,
                                              const std::vector<NDArray*>& outputs) {
  using namespace nnvm;
  using namespace imperative;

  {
  auto state_ptr = GetCachedOpThreadSafeState(default_ctx);
  OpStatePtr op_state;
  //auto op_state = OpStatePtr::Create<DynamicRuntime>();
  //auto &runtime = op_state.get_state<DynamicRuntime>();
  {
    auto &state = state_ptr.get_state<CachedOpThreadSafeState>();
    std::lock_guard<std::mutex> lock(state.mutex);
    SetForwardGraph(&state.info, inputs);
    info.fwd_graph = state.info.fwd_graph;
  }
  nnvm::Graph &g = info.fwd_graph;
  const auto &idx = g.indexed_graph();
  size_t num_inputs = idx.input_nodes().size();
  size_t max_nodes = info.fwd_graph.indexed_graph().num_nodes();
  op_states.resize(max_nodes);
  auto &states = op_states;

  // Allocate entries
  buff.resize(idx.num_node_entries());
  states.resize(idx.num_nodes());
  std::vector<NDArray *> arrays;
  arrays.reserve(buff.size());
  for (auto &buffered_array : buff) {
    arrays.push_back(&buffered_array);
  }
  for (size_t i = 0; i < num_inputs; ++i) {
    arrays[idx.entry_id(idx.input_nodes()[i], 0)] = inputs[i];
  }
  for (size_t i = 0; i < idx.outputs().size(); ++i) {
    auto eid = idx.entry_id(idx.outputs()[i]);
    if (!arrays[eid]->is_none())
      *outputs[i] = arrays[eid]->Detach();
    arrays[eid] = outputs[i];
  }
  // Allocate NDArrays
  std::vector<uint32_t> ref_count = g.GetAttr<std::vector<uint32_t>>(
      "forward_ref_count");

  std::vector<OpReqType> array_reqs(arrays.size(), kWriteTo);
  for (size_t i = 0; i < idx.num_node_entries(); ++i) {
    if (ref_count[i] == 0)
      array_reqs[i] = kNullOp;
  }
  const auto &dispatch_modes = g.GetAttr<DispatchModeVector>("dispatch_mode");
  const auto &mem_plan = g.GetAttr<MemoryPlanVector>("forward_mem_plan");
  AllocateMemory(g, idx, default_ctx, 0, idx.num_node_entries(), mem_plan,
                 arrays, &array_reqs);
  const auto &dtypes = g.GetAttr<DTypeVector>("dtype");
  const auto &shapes = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto &stypes = g.GetAttr<StorageTypeVector>("storage_type");
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto eid = idx.entry_id(idx.outputs()[i]);
    arrays[eid] = outputs[i];
    if (!outputs[i]->is_none())
      continue;
    *outputs[i] = NDArray(static_cast<NDArrayStorageType>(stypes[eid]),
                          shapes[eid], default_ctx, true, dtypes[eid]);
  }
  // If CachedOp is running in the inline mode, it uses RunGraph to record
  // computation; otherwise, CachedOp records computation itself.
  // So if it's not the inline mode, we disable recording.
  RunGraph(false, idx, arrays, 0, idx.num_nodes(), std::move(array_reqs),
           std::move(ref_count), &states, dispatch_modes, false);
  return op_state;
  }
}

OpStatePtr CachedOpThreadSafe::Forward(const std::shared_ptr<CachedOpThreadSafe>& op_ptr,
                                       const std::vector<NDArray*>& inputs,
                                       const std::vector<NDArray*>& outputs) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_EQ(inputs.size(), num_inputs());
  Context default_ctx = inputs[0]->ctx();
  const auto& idx = fwd_graph_.indexed_graph();
  for (size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs[i]->ctx(), default_ctx)
        << "CachedOp requires all inputs to live on the same context. But "
        << idx[idx.input_nodes()[0]].source->attrs.name
        << " is on " << default_ctx << " while "
        << idx[idx.input_nodes()[i]].source->attrs.name
        << " is on " << inputs[i]->ctx();
  }

  OpStatePtr op_state;
  try {
    op_state = DynamicForward(default_ctx, inputs, outputs);
  } catch (const dmlc::Error& e) {
    throw e;
  }
  return op_state;

}

struct CachedOpThreadSafeActualState {
  std::shared_ptr<CachedOpThreadSafe> op;
  OpStatePtr forward_state;

  explicit CachedOpThreadSafeActualState(std::shared_ptr<CachedOpThreadSafe> op) {
    this->op = op;
  }
};
OpStatePtr CreateCachedOpThreadSafeState(const NodeAttrs& attrs,
                               Context ctx,
                               const mxnet::ShapeVector& in_shapes,
                               const std::vector<int>& in_types) {
  const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
  return OpStatePtr::Create<CachedOpThreadSafeActualState>(op);
}

void CachedOpThreadSafeForward(const OpStatePtr& state_ptr,
                               const OpContext& ctx,
                               const std::vector<NDArray>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<NDArray>& outputs) {
  CachedOpThreadSafeActualState &s = state_ptr.get_state<CachedOpThreadSafeActualState>();
  std::vector<NDArray> in_bufs = inputs;
  std::vector<NDArray> out_bufs = outputs;
  std::vector<NDArray *> in_ptrs(in_bufs.size());
  std::vector<NDArray *> out_ptrs(out_bufs.size());
  for (size_t i = 0; i < in_ptrs.size(); i++)
    in_ptrs[i] = &in_bufs[i];
  for (size_t i = 0; i < out_ptrs.size(); i++)
    out_ptrs[i] = &out_bufs[i];

  // Set is_recording correct for the imperative executor.
  CHECK(!ctx.need_grad) << "Only inference use case supported with thread safe cached op";
  CHECK(!ctx.is_train) << "Only inference use case supported with thread safe cached op";
  s.forward_state = s.op->Forward(nullptr, in_ptrs, out_ptrs);
  // The arrays in out_ptrs may be changed by CachedOp.
  // If it is, we need to copy data back.
  for (size_t i = 0; i < out_bufs.size(); i++)
    if (!out_bufs[i].IsSame(outputs[i]))
      CopyFromTo(out_bufs[i], outputs[i]);
}

void CachedOpThreadSafeParamParser(nnvm::NodeAttrs* attrs) {
  CachedOpThreadSafeConfig param;
  try {
    param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
}
CachedOpThreadSafe::~CachedOpThreadSafe() {}

NNVM_REGISTER_OP(_CachedOpThreadSafe)
.set_num_inputs([](const NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op->num_inputs();
  })
.set_num_outputs([](const NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op->num_outputs();
  })
.set_attr_parser(CachedOpThreadSafeParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op->ListForwardInputNames();
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op->ListForwardOutputNames();
  })
.set_attr<FCreateOpState>("FCreateOpState", CreateCachedOpThreadSafeState)
.set_attr<mxnet::FInferShape>("FInferShape",
  [](const nnvm::NodeAttrs& attrs,
     mxnet::ShapeVector *in_shapes,
     mxnet::ShapeVector *out_shapes) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpShapeHelper(op->GetForwardSym(), in_shapes, out_shapes);
  })
.set_attr<nnvm::FInferType>("FInferType",
  [](const nnvm::NodeAttrs& attrs,
     std::vector<int> *in_types,
     std::vector<int> *out_types) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpTypeHelper(op->GetForwardSym(), in_types, out_types);
  })
.set_attr<FInferStorageType>("FInferStorageType",
  [](const nnvm::NodeAttrs& attrs,
     const int dev_mask,
     DispatchMode* dispatch_mode,
     std::vector<int>* in_stypes,
     std::vector<int>* out_stypes) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpStorageTypeHelper(op->GetForwardSym(),
                                                  dev_mask, dispatch_mode,
                                                  in_stypes, out_stypes);
  })
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<cpu>", CachedOpThreadSafeForward)
.set_attr<FStatefulComputeEx>("FStatefulComputeEx<gpu>", CachedOpThreadSafeForward)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpMutableInputsHelper(op->GetForwardSym());
  })
.set_attr<FResourceRequest>("FResourceRequest",
  [](const nnvm::NodeAttrs& attrs) {
    const CachedOpThreadSafePtr& op = nnvm::get<CachedOpThreadSafePtr>(attrs.parsed);
    return op::DefaultSubgraphOpResourceRequestHelper(op->GetForwardSym());
  })
.set_attr<FExecType>("FExecType", op::DefaultSubgraphOpExecType)
.add_argument("data", "NDArray-or-Symbol[]", "input data list");




} // namespace mxnet
