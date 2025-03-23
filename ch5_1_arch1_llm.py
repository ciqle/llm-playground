from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

model = ChatOllama(base_url="http://localhost:11434", model="qwen2.5:32b")


class State(TypedDict):
    # Messages have the type "list". The `add_messages`
    # function in the annotation defines how this state should
    # be updated (in this case, it appends new messages to the
    # list, rather than replacing the previous messages)
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


if __name__ == "__main__":
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    # Build the graph like previous chatpers
    graph = builder.compile()

    input = {"messages": [HumanMessage("hi!")]}
    for chunk in graph.stream(input):
        print(chunk)

# Output:
# {
#     "chatbot": {
#         "messages": [
#             AIMessage(
#                 content="Hello! How can I assist you today? Feel free to ask me any questions or let me know if you need help with anything specific.",
#                 additional_kwargs={},
#                 response_metadata={
#                     "model": "qwen2.5:32b",
#                     "created_at": "2025-03-23T08:36:13.43025Z",
#                     "done": True,
#                     "done_reason": "stop",
#                     "total_duration": 2720853375,
#                     "load_duration": 668334041,
#                     "prompt_eval_count": 31,
#                     "prompt_eval_duration": 656000000,
#                     "eval_count": 29,
#                     "eval_duration": 1391000000,
#                     "message": Message(
#                         role="assistant", content="", images=None, tool_calls=None
#                     ),
#                 },
#                 id="run-e1ce131c-1262-49d1-85d7-4167188eb306-0",
#                 usage_metadata={
#                     "input_tokens": 31,
#                     "output_tokens": 29,
#                     "total_tokens": 60,
#                 },
#             )
#         ]
#     }
# }
