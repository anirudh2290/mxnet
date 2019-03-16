#ifndef MXNET_OPERATOR_SUBGRAPH_ADDITIONAL_H_
#define MXNET_OPERATOR_SUBGRAPH_ADDITIONAL_H_

#include "./common.h"
#include "../../imperative/cached_op.h"
#include "./subgraph_property.h"
#include <dmlc/parameter.h>
#include <nnvm/symbolic.h>
#include <nnvm/graph.h>
#include <vector>
#include <string>

namespace mxnet {
namespace op {

/*
 * This selects nodes for a subgraph that only contains operators
 * in a given set and it visits nodes via both input and output links.
 */
class TestContainOpSelector: public SubgraphSelector {
 public:
  explicit TestContainOpSelector(const std::unordered_set<std::string>& op_names)
    : op_names_(op_names) {}

  virtual bool Select(const nnvm::Node &seed_node) {
    return !seed_node.is_variable() && op_names_.count(seed_node.op()->name);
  }

  virtual bool SelectInput(const nnvm::Node &cur_node, const nnvm::Node &input_node) {
    return !input_node.is_variable() && op_names_.count(input_node.op()->name);
  }

  virtual bool SelectOutput(const nnvm::Node &cur_node, const nnvm::Node &output_node) {
    return !output_node.is_variable() && op_names_.count(output_node.op()->name);
  }
 private:
  const std::unordered_set<std::string>& op_names_;
};

/*
 * This subgraph property finds a subgraph whose nodes have only operators
 * within a set. The operators in the subgraph will be executed by _CachedOp.
 */
class TestDefaultSubgraphProperty: public SubgraphProperty {
 public:
  static SubgraphPropertyPtr Create() { return std::make_shared<TestDefaultSubgraphProperty>(); }
  virtual nnvm::NodePtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                           const int subgraph_id = 0) const {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_CachedOp");
    n->attrs.name = "_CachedOp" + std::to_string(subgraph_id);
    n->attrs.subgraphs.push_back(std::make_shared<nnvm::Symbol>(sym));

    std::vector<std::pair<std::string, std::string> > flags{{"static_alloc", "true"}};
    n->attrs.parsed = CachedOpPtr(new CachedOp(sym, flags));

    return n;
  }
  virtual SubgraphSelectorPtr CreateSubgraphSelector() const {
    return std::make_shared<TestContainOpSelector>(
        this->GetAttr<std::unordered_set<std::string>>("op_names"));
  }
};


MXNET_REGISTER_SUBGRAPH_PROPERTY(test, TestDefaultSubgraphProperty);

}
}

#endif
