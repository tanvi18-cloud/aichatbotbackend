from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from instructions import information

# ✅ Directly adding API key instead of using .env
OPENAI_API_KEY = "your_openai_api_key_here"

# Define a custom fallback tool for unknown queries
@tool(description="Called when query falls outside your knowledge base")
def web_search(query: str) -> str:
    return "This information is not publicly available right now."

tools = [web_search]

# ✅ Initialize LangChain AI Client
client = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.4, 
    api_key="YOUR_API_KEY_HERE" # Using the direct API key
)

# ✅ Define chatbot prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", information),
    ("placeholder", "{messages}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ✅ Create chatbot agent
agent = create_tool_calling_agent(client, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ✅ Temporary chat history storage
demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

conversational_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="messages",
    output_messages_key="output",
)

# ✅ Function to process user messages
def run_agent(messages, session_id):
    response = conversational_agent_executor.invoke(
        {"messages": [messages]},
        {"configurable": {"session_id": session_id}},
    )   
    return response
