from dotenv import load_dotenv

from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from nodes import(
    safety_check_node,
    agent_reasoning_node,
    tool_node
)

load_dotenv()


## NODE NAMES

AGENT_REASON='agent_reason'
SAFETY='safety'
ACT='act'
LAST=-1

def check_safety(state:MessagesState)-> str:
    """
    After safety check runs, decide whether to continue or stop
    """

    if "cannot" in state["messages"][LAST].content.lower() or "hack" in state['messages'][LAST].content.lower():
        return END
    return AGENT_REASON

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


# GRAPH BUILD 

flow = StateGraph(MessagesState)

# add nodes
flow.add_node(SAFETY,safety_check_node)
flow.add_node(AGENT_REASON,agent_reasoning_node)
flow.add_node(ACT,tool_node)

flow.set_entry_point(SAFETY)

# Conditional edges 
flow.add_conditional_edges(
    SAFETY,
    check_safety,
    {
        END:END,
        AGENT_REASON:AGENT_REASON
    }
)

flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        END: END,
        ACT: ACT
    }
)

flow.add_edge(ACT,AGENT_REASON)

app=flow.compile()

app.get_graph().draw_mermaid_png(output_file_path="Practice/flow.png")

if __name__=="__main__":
    res = app.invoke({"messages":[HumanMessage(content="What is the temperature in Hyderabad in India? List it and triple it.")]})
    print(res["messages"][LAST].content)