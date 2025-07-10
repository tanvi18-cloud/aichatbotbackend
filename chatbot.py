import os
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from instructions import information

# ✅ Fallback Tool
@tool(description="Called when query falls outside your knowledge base")
def web_search(query: str) -> str:
    return "This information is not publicly available right now."

tools = [web_search]

# ✅ Load API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_API_KEY_HERE")

# ✅ LangChain Client
client = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    api_key=GROQ_API_KEY
)

# ✅ Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", information),
    ("placeholder", "{messages}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ✅ Agent
agent = create_tool_calling_agent(client, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ✅ Chat history
demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="messages",
    output_messages_key="output",
)

# ✅ Main function
def run_agent(messages, session_id):
    response = conversational_agent_executor.invoke(
        {"messages": [messages]},
        {"configurable": {"session_id": session_id}},
    )
    return response
