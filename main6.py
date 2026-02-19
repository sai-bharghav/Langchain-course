# Load environment variables from .env file
# (This is usually where API keys like OpenAI key are stored)
from dotenv import load_dotenv

# HumanMessage is used when sending user messages into the graph
from langchain_core.messages import HumanMessage

# MessagesState → prebuilt state schema that stores conversation messages
# StateGraph → used to build the workflow graph
# END → special constant that tells the graph to stop execution
from langgraph.graph import MessagesState, StateGraph, END

# Import two nodes defined in another file
# run_agent_reasoning → calls the LLM
# tool_node → executes tools when requested
from nodes6 import tool_node, run_agent_reasoning


# Load environment variables
load_dotenv()


# ----------------------------------------
# Define Node Names (just string labels)
# ----------------------------------------

# Node name for LLM reasoning step
AGENT_REASON = "agent_reason"

# Node name for tool execution step
ACT = "act"

# Index to get the last message in the messages list
LAST = -1


# ----------------------------------------
# Conditional Function
# ----------------------------------------

def should_continue(state: MessagesState) -> str:
    """
    This function decides what happens AFTER the LLM reasoning step.

    It checks the last message in the state.
    If the LLM requested a tool → go to ACT node.
    If not → stop the graph (END).
    """

    # Get the last message from conversation
    # If it does NOT contain any tool calls,
    # it means LLM is done answering.
    if not state["messages"][LAST].tool_calls:
        return END  # Stop execution

    # If there ARE tool calls, go execute them
    return ACT


# ----------------------------------------
# Build the Graph
# ----------------------------------------

# Create a graph that uses MessagesState as shared memory
flow = StateGraph(MessagesState)


# Add the reasoning node (LLM call)
flow.add_node(AGENT_REASON, run_agent_reasoning)

# Define where execution starts
flow.set_entry_point(AGENT_REASON)

# Add tool execution node
flow.add_node(ACT, tool_node)


# ----------------------------------------
# Add Conditional Routing
# ----------------------------------------

# After AGENT_REASON runs,
# call should_continue() to decide next step.
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        END: END,   # If should_continue returns END → stop
        ACT: ACT    # If should_continue returns ACT → go to tool node
    }
)


# After tool execution,
# go back to reasoning step.
# This creates the ReAct loop:
# Reason → Act → Reason → Act → ...
flow.add_edge(ACT, AGENT_REASON)


# ----------------------------------------
# Compile the Graph
# ----------------------------------------

# Compile converts the graph definition into an executable app
app = flow.compile()


# Optional: visualize the graph structure as an image
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


# ----------------------------------------
# Main Execution
# ----------------------------------------

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
