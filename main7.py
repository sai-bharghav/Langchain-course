# Changing the LANGCHAIN_PROJECT= Reflection Agent
from dotenv import load_dotenv

load_dotenv()


from langchain_core.messages import BaseMessage, HumanMessage
# BaseMessage = “Any chat message”
# HumanMessage = “User message specifically”
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
# When updating the messages list, append new messages instead of overwriting

from chains7 import generation_chain, reflection_chain
from typing import TypedDict, Annotated


#“My graph state has one thing called messages.
# It is a list of chat messages.
# Whenever a node adds a message, append it instead of overwriting.”
class MessageGraph(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(State:MessageGraph):
    return {
        "messages":[generation_chain.invoke({"messages":State["messages"]})]
    }

def reflection_node(State:MessageGraph):
    result = reflection_chain.invoke({"messages":State["messages"]})
    return {
        "messages":[HumanMessage(content=result.content)]
    }


builder = StateGraph(state_schema=MessageGraph)

builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

builder.set_entry_point(GENERATE)

def should_continue(state:MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(
    GENERATE,
    should_continue,{
        REFLECT:REFLECT,
        END:END
    }
)

builder.add_edge(REFLECT, GENERATE)

graph=builder.compile()
print(graph.get_graph().draw_mermaid())
if __name__=="__main__":
    print('Hello Reflection Agent')

    inputs = HumanMessage(content=""" Make this tweet better and more viral:
        OpenClaw had already shipped 100 fixes and moved on.

Security hardened. Gemini 3.1 in. Discord voice live. iOS stable. Prompt caching improved. All dropped in one silent update while the industry was busy with definitions.

The most dangerous teams in any market are the ones that build in silence and let the changelog speak. No hype, no debates just code that ships.

When the smartest people in the room are still talking theory, the ones executing quietly are already three steps ahead.
""")
    result = graph.invoke({"messages":[inputs]})
    print(result)