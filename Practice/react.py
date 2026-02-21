from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

@tool
def triple(num:float)-> float:
    """
    The agent can use this tool when there is a request to triple the number

    Params

        Args: num- The number it gets and should triple
        returns: The number should be tripled
    """
    return num*3

tools=[TavilySearch(), triple]

llm = ChatOpenAI(model='gpt-4o',temperature=0).bind_tools(tools)