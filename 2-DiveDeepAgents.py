from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
# ReActSingleInputOutputParser is now in langchain_classic to maintain stable legacy logic
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from typing import Union, List

load_dotenv()

# Helper function to find the right tool object based on the string name returned by the LLM
def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool 
    raise ValueError(f'Tool with name {tool_name} not found')

# The @tool decorator automatically creates a Tool object. 
# The docstring below becomes the 'description' the LLM reads to decide when to use it.
@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters."""
    print(f'get_text_length received text: {text}')

    # LLMs sometimes wrap tool inputs in quotes or add newlines. 
    # This 'sanitizes' the input so we only count the actual characters.
    text = text.strip("'\n").strip('"')
    return len(text)

def format_log_to_str(
        intermediate_steps: List[tuple[AgentAction, str]],
        observation_prefix: str= "Observation: ",
        llm_prefix:str = "Thought: "
)->str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts=""
    for action,observation in intermediate_steps:
        thoughts+=action.log
        thoughts+= f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts

if __name__ == '__main__':
    print("Hello reAct LangChain!!")
    tools = [get_text_length]

    # This template defines the 'Reasoning' framework (ReAct).
    # It forces the LLM to write its thoughts before taking an action.
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat 1 time)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    # .partial() fills in the tool-related variables immediately so the LLM knows its 'capabilities'.
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), 
        tool_names=", ".join(t.name for t in tools)
    )

    # model_kwargs "stop" sequences are CRITICAL.
    # We tell the LLM to STOP generating text as soon as it writes "Observation:".
    # This gives control back to our Python code so WE can provide the tool result.
    llm = ChatOpenAI(
        temperature=0, 
        model='gpt-4o', 
        model_kwargs={"stop": ["\nObservation", "Observation:"]}
    )
    intermediate_steps=[]

    # THE AGENT CHAIN:
    # 1. Takes 'input' string.
    # 2. Formats the ReAct prompt.
    # 3. LLM generates the "Thought" and "Action".
    # 4. Parser turns that text into an AgentAction or AgentFinish object.
    agent = (
        {
            "input": lambda x: x['input'],
            "agent_scratchpad": lambda x: format_log_to_str(x['agent_scratchpad'])
        } 
    | prompt 
    | llm 
    | ReActSingleInputOutputParser()
    )

    # Step 1: The agent reasons and decides which tool to use.
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of the text 'Hello, world!'?",
                                                                "agent_scratchpad": intermediate_steps})
    
    print(agent_step)
    
    # agent_step.log contains the full 'Thought' and 'Action' text from the LLM.
    print(agent_step.log)

    # If the parser returned an 'AgentAction', it means the LLM wants to use a tool.
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        # We manually execute the tool function.
        observation = tool_to_use.func(str(tool_input))
        
        # We print the result, which in a real loop would be fed back into the LLM 
        # for its next 'Thought' step.
        print(f'Observation: {observation}')
        intermediate_steps.append((agent_step,observation))