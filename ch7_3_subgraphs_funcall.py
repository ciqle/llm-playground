from typing import TypedDict

from langgraph.graph import START, StateGraph

"""
与ch7_2_subgraphs_direct.py的主要区别:
1. 这个文件展示了函数式调用子图的方法 - 即在普通函数中调用子图
2. 子图不会被直接添加到父图的node中,而是在父图的节点函数内被invoke
3. 这种方式更灵活，能在函数中添加更多处理逻辑，并且不会在父图中直接暴露子图节点
4. 架构上更清晰地分离了子图和主图的逻辑
"""


class State(TypedDict):
    foo: str


class SubgraphState(TypedDict):
    # 这些键都不与父图状态共享
    bar: str
    baz: str


# 定义子图
def subgraph_node(state: SubgraphState) -> SubgraphState:
    return {"bar": state["bar"] + "baz"}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)
subgraph_builder.add_edge(START, "subgraph_node")
# 这里可以添加子图的其他设置

# 先把subgraph给编译出来，这样后面可以直接调用
subgraph = subgraph_builder.compile()


# 定义调用子图的父图节点
def node(state: State) -> State:
    # 将父图状态转换为子图状态
    response = subgraph.invoke({"bar": state["foo"]})
    # 将响应转换回父图状态
    return {"foo": response["bar"]}


# 注意：在这种方法中，subgraph作为一个可调用对象在node函数中被调用
# 而不是被直接添加到父图的节点中

builder = StateGraph(State)
# 注意这里我们使用的是函数节点而不是编译好的子图
builder.add_node("node", node)
builder.add_edge(START, "node")
# 这里可以添加父图的其他设置
graph = builder.compile()


if __name__ == "__main__":
    # 使用示例
    initial_state = {"foo": "hello"}
    result = graph.invoke(initial_state)
    print(f"Result: {result}")  # 应该将foo转换为bar，附加"baz"，然后将bar转换回foo
    # Result: {'foo': 'hellobaz'}
