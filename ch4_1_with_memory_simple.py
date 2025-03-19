from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 创建一个简单的提示模板，包含系统指令和消息占位符
# 注意：这里使用了 "placeholder" 来处理消息历史，这不是标准的 ChatPromptTemplate 格式
# 使用 placeholder 可能会导致消息历史的角色信息(human/ai)丢失，因为它只是简单地将消息内容插入
# 更好的做法是使用 ChatPromptTemplate 的标准格式，即明确的 human/ai 角色标记
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        ("placeholder", "{messages}"),
    ]
)

model = ChatOllama(
    model="qwen2.5:32b",
    base_url="http://localhost:11434",
)

chain = prompt | model

if __name__ == "__main__":

    # 这里填充了历史消息
    # 问题：当我们使用 placeholder 和直接传入历史消息列表时，
    # LangChain 不会正确处理消息的角色信息，可能导致上下文关系不清晰
    # 应该使用 LangChain 的 memory 组件来正确管理对话历史
    res = chain.invoke(
        {
            "messages": [
                (
                    "human",
                    "Translate this sentence from English to French: I love programming.",
                ),
                ("ai", "J'adore programmer."),
                ("human", "What did you just say?"),
            ],
        }
    )
    print(res)

# LLM Response Object, don't run this code
response = {
    "content": 'I translated "I love programming" into French, which is "J\'adore programmer."',
    "additional_kwargs": {},
    "response_metadata": {
        "model": "qwen2.5:32b",
        "created_at": "2025-03-19T16:28:14.887331Z",
        "done": True,
        "done_reason": "stop",
        "total_duration": 1438873125,  # Duration in nanoseconds
        "load_duration": 27944375,  # Model loading time in nanoseconds
        "prompt_eval_count": 63,
        "prompt_eval_duration": 471000000,
        "eval_count": 20,
        "eval_duration": 936000000,
        "message": {
            "role": "assistant",
            "content": "",
            "images": None,
            "tool_calls": None,
        },
    },
    "id": "run-58feda1e-44c7-4d21-9a5e-afac377099bb-0",
    "usage_metadata": {"input_tokens": 63, "output_tokens": 20, "total_tokens": 83},
}
