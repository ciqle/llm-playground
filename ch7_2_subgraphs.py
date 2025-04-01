from typing import TypedDict

from langgraph.graph import START, StateGraph


class State(TypedDict):
    foo: str


class SubgraphState(TypedDict):
    # 这些键与父图状态不共享
    bar: str
    baz: str


# 定义子图
def subgraph_node(state: SubgraphState) -> SubgraphState:
    return {"bar": state["bar"] + "baz"}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)
subgraph_builder.add_edge(START, "subgraph_node")
# 这里可以添加更多子图设置
subgraph = subgraph_builder.compile()


# 定义调用子图的父图节点
def node(state: State) -> State:
    # 将父图状态转换为子图状态
    # 将父图里的foo传递给子图的bar
    response = subgraph.invoke({"bar": state["foo"]})
    # 将响应转换回父图状态
    return {"foo": response["bar"]}


builder = StateGraph(State)
# 注意我们使用的是`node`函数而不是编译后的子图
builder.add_node("node", node)
builder.add_edge(START, "node")
# 这里可以添加更多父图设置
graph = builder.compile()

# 使用示例
if __name__ == "__main__":
    # 定义初始状态
    initial_state = {"foo": "hello"}
    # 调用图
    result = graph.invoke(initial_state)
    print(f"Result: {result}")  # 应该将foo转换为bar，添加"baz"，然后将bar转换回foo
    # Result: {'foo': 'hellobaz'}
