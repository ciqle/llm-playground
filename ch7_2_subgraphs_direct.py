from typing import TypedDict

from langgraph.graph import START, StateGraph


# 为父图和子图定义状态类型
class State(TypedDict):
    foo: str  # 这个键与子图共享


class SubgraphState(TypedDict):
    foo: str  # 这个键与父图共享
    bar: str


# 定义子图
def subgraph_node(state: SubgraphState) -> SubgraphState:
    # 注意这个子图节点可以通过共享的"foo"键与父图通信
    return {"foo": state["foo"] + "bar"}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)
# 这里可以添加更多子图设置
subgraph_builder.add_edge(START, "subgraph_node")
subgraph = subgraph_builder.compile()

# 定义父图
builder = StateGraph(State)
builder.add_node("subgraph", subgraph)
builder.add_edge(START, "subgraph")
# 这里可以添加更多父图设置
graph = builder.compile()

if __name__ == "__main__":
    # 使用示例
    initial_state = {"foo": "hello"}
    result = graph.invoke(initial_state)
    print(f"Result: {result}")
    # Result: {'foo': 'hellobar'}
