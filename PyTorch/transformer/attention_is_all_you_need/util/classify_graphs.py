import argparse
import re
import torch

from enum import Enum

class OpType(Enum):
    POINTWISE = 1
    NORMS = 2
    REDUCTIONS = 3
    VIEWS_EXPANDS = 4
    REMOVE = 5
    IGNORE = 6

op_types = {
    "aten::rsqrt": OpType.POINTWISE,
    "aten::abs": OpType.POINTWISE,
    "aten::eq": OpType.POINTWISE,
    "aten::gelu": OpType.POINTWISE,
    "aten::remainder": OpType.POINTWISE,
    "aten::_softmax": OpType.POINTWISE,
    "aten::clamp": OpType.POINTWISE,
    "aten::gt": OpType.POINTWISE,
    "aten::mul": OpType.POINTWISE,
    "aten::add": OpType.POINTWISE,
    "aten::sum": OpType.REDUCTIONS,
    "aten::ne": OpType.POINTWISE,
    "aten::silu": OpType.POINTWISE,
    "aten::pow": OpType.POINTWISE,
    "aten::ge": OpType.POINTWISE,
    "aten::native_batch_norm": OpType.NORMS,
    "aten::sub": OpType.POINTWISE,
    "aten::mean": OpType.REDUCTIONS,
    "aten::sqrt": OpType.POINTWISE,
    "aten::reciprocal": OpType.POINTWISE,
    "aten::reshape": OpType.VIEWS_EXPANDS,
    "aten::relu": OpType.POINTWISE,
    "prim::Constant": OpType.REMOVE,
    "prim::TupleConstruct": OpType.IGNORE,
    "aten::div": OpType.POINTWISE,
    "aten::tanh": OpType.POINTWISE,
    "aten::neg": OpType.POINTWISE,
    "aten::log": OpType.POINTWISE,
    "aten::unsqueeze": OpType.VIEWS_EXPANDS,
    "aten::native_layer_norm": OpType.NORMS,
    "aten::exp": OpType.POINTWISE,
    "aten::sigmoid": OpType.POINTWISE,
}

def type_to_placeholder(op_type: OpType) -> str:
    mapping = {
        OpType.POINTWISE: "aten::pointwise_placeholder",
        OpType.NORMS: "aten::norm_placeholder",
        OpType.REDUCTIONS: "aten::reduction_placeholder",
        OpType.VIEWS_EXPANDS: "aten::view_expand_placeholder",
        OpType.IGNORE: "aten::ignore_placeholder",
        OpType.REMOVE: "aten::remove_placeholder",
    }
    return mapping[op_type]


# get the op type. op_name is expected to be the qualified name.
def get_type(op_name: str) -> OpType:
    if op_name in op_types:
        return op_types[op_name]
    for optype in OpType:
        if type_to_placeholder(optype) == op_name:
            return optype
    raise NotImplementedError(f"No OpType known for op '{op_name}'")


def simplify_tensor_type(jit_type):
    if isinstance(jit_type, torch._C.TensorType):
        return torch._C.TensorType.get()
    return jit_type


def remove_inputs(graph):
    inputs_size = 0
    for n in graph.inputs():
        inputs_size += 1
        for use in n.uses():
            use.user.removeInput(use.offset)

    for i in reversed(range(inputs_size)):
        graph.eraseInput(i)

    return graph


# Remove vertices like x or y below, where x or y are pointwise.
#   (pointwise) --> (x) --> (...)
#   (...) --> (y) --> (pointwise)
# if remove_all is true, then it doesn't care if pointwise ops preceed/succeed x or y.
def remove_duplicate_pointwise(graph, remove_all=False):
    to_remove = []
    old_str = str(graph)
    def bypass_node(n):
        to_remove.append(n)
        n.output().replaceAllUsesWith(n.input())

    for n in graph.nodes():
        if get_type(n.kind()) != OpType.POINTWISE:
            continue
        if n.inputsSize() != 1 or n.outputsSize() != 1:
            continue
        if get_type(n.input().node().kind()) == OpType.POINTWISE or remove_all:
            bypass_node(n)
            continue
        uses = [r.user for r in n.output().uses() if r.user.kind() != "prim::Return"]
        if len(uses) >= 1 and (all(get_type(r.kind()) == OpType.POINTWISE for r in uses) or remove_all):
            bypass_node(n)
            continue

    for n in reversed(to_remove):
        n.destroy()

    return graph


def compress_graph(graph):
    old_nodes = []
    erased_nodes = set()
    for n in graph.nodes():
        simple_type = get_type(n.kind())
        if simple_type == OpType.IGNORE:
            continue
        old_nodes.append(n)
        if simple_type == OpType.REMOVE:
            erased_nodes.add(n)
            continue
        new_node = graph.create(type_to_placeholder(simple_type), n.outputsSize())
        new_node.insertBefore(n)
        for inp in n.inputs():
            if inp.node() not in erased_nodes:
                new_node.addInput(inp)
        for old_out, new_out in zip(n.outputs(), new_node.outputs()):
            new_out.setType(simplify_tensor_type(old_out.type()))
            old_out.replaceAllUsesWith(new_out)
    for n in reversed(old_nodes):
        n.destroy()
    graph = remove_inputs(graph)
    graph = remove_duplicate_pointwise(graph)
    return torch._C._jit_pass_canonicalize(graph, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
Collection of helper functions for eliminating duplicate subgraphs

Usage:
~~~
import classify_graphs

# some ir string called "ir"
graph = torch._C.parse_ir(ir)

# "hashes" the graph based on categories of ops (pointwise, reductions, views/expands, norms)
compressed_graph = classify_graphs.compress_graph(graph)

# do something with the compressed graph
~~~

Alternatively, call it and it will return one graph per hashed category
Usage:
python3 log_extract.py log.txt --output > log_result.py
python3 classify_graphs.py log_result.py > filtered_logs.py
""", formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filename", type=str, help="output from log_extract.py --help")
    args = parser.parse_args()
    with open(args.filename) as f:
        arr = eval(f.read())

    # see 73984
    for i in range(len(arr)):
        if len(re.findall(r'value=annotate\(List\[int', arr[i])) >= 1:
            arr[i] = arr[0]

    classified = {}
    for ir in arr:
        graph = torch._C.parse_ir(ir)
        graph = compress_graph(graph)
        graph_class = str(graph)
        if graph_class not in classified:
            classified[graph_class] = []
        classified[graph_class].append(ir)

    final_selection = []
    for cl, graphs in classified.items():
        # choose the longest graph of this type
        s = sorted(graphs, key=lambda x: -len(str(x)))
        final_selection.append(str(graphs[0]))

    print('[' + ', '.join(f'"""{x}"""' for x in final_selection) + ']')
