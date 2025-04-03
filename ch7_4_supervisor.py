import functools
import logging
import time
from typing import Literal

from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel

# TODO: Still has bugs to be fixed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("agent_workflow.log")],
)
logger = logging.getLogger("agent_workflow")


# Decorator for logging agent function calls
def log_agent_call(func):
    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
        agent_name = func.__name__
        logger.info(f"Starting {agent_name} agent")
        start_time = time.time()
        result = func(state, *args, **kwargs)
        end_time = time.time()
        logger.info(f"Completed {agent_name} agent in {end_time - start_time:.2f}s")

        # Log the result if appropriate
        if "next" in result:
            logger.info(f"{agent_name} decided next step: {result['next']}")
        if "messages" in result and result["messages"]:
            logger.info(
                f"{agent_name} generated response (truncated): {str(result['messages'][-1].content)[:100]}..."
            )

        return result

    return wrapper


# next字段必须从"researcher"、"coder"或"FINISH"中选择
class SupervisorDecision(BaseModel):
    next: Literal["researcher", "coder", "FINISH"]


# Initialize model
model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen2.5:32b",
    temperature=0,
)
model = model.with_structured_output(SupervisorDecision)

# Define available agents
agents = ["researcher", "coder"]

# Define system prompts

# （给supervisor的prompt）
# 你是一名主管，负责管理以下工人之间的对话：
# 研究员和编码员。根据以下用户请求，决定下一个行动的工人。
# 每个工人将执行一个任务并回复他们的结果和状态。
# 完成后，回复FINISH。
system_prompt_part_1 = f"""You are a supervisor tasked with managing a conversation between the  
following workers: {agents}. Given the following user request,  
respond with the worker to act next. Each worker will perform a  
task and respond with their results and status. When finished,  
respond with FINISH."""

# 根据前面的对话，谁应该下一个行动？或者我们应该FINISH吗？选择一个：researcher、coder、FINISH
system_prompt_part_2 = f"""Given the conversation above, who should act next? Or should we FINISH? Select one of: {", ".join(agents)}, FINISH"""


@log_agent_call
def supervisor(state):
    logger.info("Supervisor analyzing conversation state")
    messages = [
        ("system", system_prompt_part_1),
        *state["messages"],
        ("system", system_prompt_part_2),
    ]
    response = model.invoke(messages)
    logger.info(f"Supervisor decision: {response.next}")
    # 返回包含next决策的字典，不更改消息
    return {"next": response.next}


# Define agent state
class AgentState(MessagesState):
    next: Literal["researcher", "coder", "FINISH"]


# Define agent functions
@log_agent_call
def researcher(state: AgentState):
    # In a real implementation, this would do research tasks
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "You are a research assistant. Analyze the request and provide relevant information.",
            },
            {"role": "user", "content": state["messages"][0].content},
        ]
    )
    return {"messages": [response]}


@log_agent_call
def coder(state: AgentState):
    # In a real implementation, this would write code
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "You are a coding assistant. Implement the requested functionality.",
            },
            {"role": "user", "content": state["messages"][0].content},
        ]
    )
    return {"messages": [response]}


# Build the graph
builder = StateGraph(AgentState)
logger.info("Building agent workflow graph")
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

builder.add_edge(START, "supervisor")
# Route to one of the agents or exit based on the supervisor's decision
builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

graph = builder.compile()
logger.info("Agent workflow graph compiled successfully")

# Example usage
if __name__ == "__main__":
    logger.info("Starting agent workflow execution")
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": "I need help analyzing some data and creating a visualization.",
            }
        ],
        "next": "supervisor",
    }

for output in graph.stream(initial_state):
    logger.info(f"Graph step output: {output}")

    # 获取当前执行的节点名称（第一个键）
    node_name = next(iter(output), None)

    if node_name:
        node_output = output[node_name]
        # 现在从节点输出中获取 next
        next_step = node_output.get("next", "N/A")
        print(f"\nStep decision: {next_step}")

        # 从节点输出中获取 messages
        if "messages" in node_output and node_output["messages"]:
            print(f"Response: {node_output['messages'][-1].content[:100]}...")

    logger.info("Agent workflow execution completed")
