from dotenv import load_dotenv

from langgraph.graph import MessagesState

from langgraph.prebuilt import ToolNode

from react import llm,tools

from langchain_core.messages import SystemMessage

load_dotenv()

SYSTEM_MESSAGE="You are an helpful assistant which helps to answer the questions using tools if needed"
SYSTEM_CHECK_MESSAGE="You are an assistant that you will check whether the message is ethical or not."


### SAFETY CHECKING NODE

def safety_check_node(state:MessagesState)->MessagesState:
    """
    This node is responsible for judging whether the message

    It:
    1. Takes the current state (which includes messages)
    2. Adds a system check message
    3. Calls the LLM
    4. Return the same message in the state if the message is ethical, right and do not have any legal issues with the message, if it not then it should respond accordingly that it cannot answer it.
    """

    response = llm.invoke(
        [
            {'role':'system',
            'content':SYSTEM_CHECK_MESSAGE},
        *state['messages']
        ]
    )

    return {"messages":[response]}


### Agent Reasoning Node

def agent_reasoning_node(state:MessagesState)-> MessagesState:
    """
    This node is responsible for calling the LLM.

    It:
    1. Takes the current state (which includes messages)
    2. Adds a system message
    3. Calls the LLM
    4. Returns updated messages
    """

    messages = state["messages"]

    # Only add system message if it's the first turn
    if not any(msg.type == "system" for msg in messages):
        messages = [SystemMessage(content=SYSTEM_MESSAGE)] + messages

    response = llm.invoke(messages)

    return {"messages": state["messages"] + [response]}


## As the homework, before the agent starts reasoning, it must first check whether the user query is safe or not 

# Tools 

tool_node = ToolNode(tools)


