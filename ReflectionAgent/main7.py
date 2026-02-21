# --------------------------------------------
# Load environment variables
# (Needed for OPENAI_API_KEY, LANGCHAIN_PROJECT, etc.)
# --------------------------------------------
from dotenv import load_dotenv
load_dotenv()


# --------------------------------------------
# Message types used in LangChain / LangGraph
# --------------------------------------------
from langchain_core.messages import BaseMessage, HumanMessage
# BaseMessage  -> Parent class for all message types (AIMessage, HumanMessage, etc.)
# HumanMessage -> Specifically represents a user message


# --------------------------------------------
# Core LangGraph imports
# --------------------------------------------
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
# add_messages:
# Instead of overwriting "messages" in state,
# new messages will be appended automatically.


# --------------------------------------------
# Import LLM chains
# generation_chain -> writes improved tweet
# reflection_chain -> critiques the tweet
# --------------------------------------------
from chains7 import generation_chain, reflection_chain


from typing import TypedDict, Annotated


# --------------------------------------------
# Define Graph State Schema
# --------------------------------------------
# This tells LangGraph:
# 1. The state has ONE key -> "messages"
# 2. It is a list of chat messages
# 3. When updated, messages should be appended (not replaced)
# --------------------------------------------
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Node names (just constants for clarity)
REFLECT = "reflect"
GENERATE = "generate"


# --------------------------------------------
# GENERATION NODE
# --------------------------------------------
# Takes the current message history
# Calls generation_chain to produce a better tweet
# Returns the new AI message to be appended to state
# --------------------------------------------
def generation_node(State: MessageGraph):
    return {
        "messages": [
            generation_chain.invoke({
                "messages": State["messages"]
            })
        ]
    }


# --------------------------------------------
# REFLECTION NODE
# --------------------------------------------
# Takes the entire conversation history
# Uses reflection_chain to critique the tweet
# The critique is converted into a HumanMessage
# so the next generation step treats it as feedback
# --------------------------------------------
def reflection_node(State: MessageGraph):
    result = reflection_chain.invoke({
        "messages": State["messages"]
    })
    return {
        "messages": [
            HumanMessage(content=result.content)
        ]
    }


# --------------------------------------------
# Build the Graph
# --------------------------------------------
builder = StateGraph(state_schema=MessageGraph)

# Add nodes
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

# Set entry point (first node executed)
builder.set_entry_point(GENERATE)


# --------------------------------------------
# Conditional stopping logic
# --------------------------------------------
# This controls the reflection loop.
# Each cycle adds:
#   - 1 generation message
#   - 1 reflection message
#
# Once message count exceeds 6, stop the loop.
# Otherwise, continue reflecting.
# --------------------------------------------
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


# Add conditional edge after GENERATE node
builder.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        REFLECT: REFLECT,
        END: END
    }
)

# After reflection, always go back to generation
builder.add_edge(REFLECT, GENERATE)


# Compile graph into executable object
graph = builder.compile()

# Print Mermaid diagram (helps visualize graph structure)
print(graph.get_graph().draw_mermaid())


# --------------------------------------------
# Run the Reflection Agent
# --------------------------------------------
if __name__ == "__main__":
    print('Hello Reflection Agent')

    # Initial user input tweet
    inputs = HumanMessage(content=""" 
    Make this tweet better and more viral:

    OpenClaw had already shipped 100 fixes and moved on.

    Security hardened. Gemini 3.1 in. Discord voice live. iOS stable. Prompt caching improved. 
    All dropped in one silent update while the industry was busy with definitions.

    The most dangerous teams in any market are the ones that build in silence and let the changelog speak. 
    No hype, no debates just code that ships.

    When the smartest people in the room are still talking theory, 
    the ones executing quietly are already three steps ahead.
    """)

    # IMPORTANT:
    # Graph expects a dictionary that matches the state schema
    # So we pass {"messages": [inputs]}
    result = graph.invoke({
        "messages": [inputs]
    })

    # Final state output (contains full message history)
    print(result)