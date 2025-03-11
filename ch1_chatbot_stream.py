from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_ollama import ChatOllama

# the building blocks

# 这里是把系统消息和用户消息组合在一起的模板
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = ChatOllama(base_url="http://localhost:11434", model="qwen2.5:32b")


@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token


if __name__ == "__main__":
    for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
        print(part)
