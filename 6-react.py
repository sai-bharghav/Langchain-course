from dotenv import load_dotenv
from langchain_core import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()

@tool
def triple(num:float) ->float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num)*3


tools= [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(temperature=0, model='gpt-4').bind_tools(tools)

