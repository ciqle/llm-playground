from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    # add_messages is a reducer function that appends the new
    # messages to the existing list
    messages: Annotated[list, add_messages]


builder = StateGraph(State)


model = ChatOllama(model="qwen2.5:32b", base_url="http://localhost:11434")


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


# This creates a node with the unique name "chatbot"
# The first argument is the unique node name
# The second argument is the function or Runnable to run

if __name__ == "__main__":

    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    graph = builder.compile()

    input = {"messages": [HumanMessage("hi!")]}

    output = graph.invoke(input)
    print(output)


# Output:
# {
#     "messages": [
#         HumanMessage(
#             content="hi!",
#             additional_kwargs={},
#             response_metadata={},
#             id="a84705af-8754-4508-b5be-36a6897e329c",
#         ),
#         AIMessage(
#             content="Hello! How can I assist you today?",
#             additional_kwargs={},
#             response_metadata={
#                 "model": "qwen2.5:32b",
#                 "created_at": "2025-03-18T15:27:18.981413Z",
#                 "done": True,
#                 "done_reason": "stop",
#                 "total_duration": 1783194500,
#                 "load_duration": 670980584,
#                 "prompt_eval_count": 31,
#                 "prompt_eval_duration": 662000000,
#                 "eval_count": 10,
#                 "eval_duration": 447000000,
#                 "message": Message(
#                     role="assistant", content="", images=None, tool_calls=None
#                 ),
#             },
#             id="run-81954682-d0b2-437c-9393-19869617d129-0",
#             usage_metadata={
#                 "input_tokens": 31,
#                 "output_tokens": 10,
#                 "total_tokens": 41,
#             },
#         ),
#     ]
# }
