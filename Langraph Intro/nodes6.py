# Load environment variables from .env file
# This is usually where your API keys (OpenAI, etc.) are stored
from dotenv import load_dotenv

# MessagesState is a prebuilt state schema from LangGraph
# It already defines a "messages" key with proper reducer behavior
from langgraph.graph import MessagesState

# ToolNode is a prebuilt node that automatically handles tool execution
# It takes tools and executes them when the LLM requests them
from langgraph.prebuilt import ToolNode

# Import your LLM instance and tools list from another file
# (Make sure the filename is valid like react6.py and NOT starting with a number)
from react6 import llm, tools


# Load environment variables into the system
load_dotenv()


# System message sets behavior for the assistant
# This tells the LLM how it should act
SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


# -----------------------------
# AGENT REASONING NODE
# -----------------------------
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    This node is responsible for calling the LLM.

    It:
    1. Takes the current state (which includes messages)
    2. Adds a system message
    3. Calls the LLM
    4. Returns updated messages
    """

    # Combine system message with conversation history
    # state["messages"] already contains user + previous messages
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_MESSAGE},
            *state["messages"]  # unpack existing conversation
        ]
    )

    # IMPORTANT:
    # We must return a dictionary that updates the state.
    # MessagesState has a reducer that APPENDS messages automatically.
    return {"messages": [response]}


# -----------------------------
# TOOL EXECUTION NODE
# -----------------------------
# ToolNode automatically:
# - Detects tool calls from LLM response
# - Executes correct tool
# - Appends tool result to messages
# - Returns updated state
tool_node = ToolNode(tools)
