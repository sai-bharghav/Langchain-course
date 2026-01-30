from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# from tavily import TavilyClient

# tavily = TavilyClient()


# A tool is a function that can be used by the agent. It can be any function we want which we can internally write by ourselves.


# The simplest way to create a tool is to use the @tool decorator. By default, the function's docstring becomes the tool's description 
# that helps the model understand when to use it:

# @tool
# def search(query:str)-> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search over internet
#     Returns:
#         The search result.
#     """
#     print(f'Searching for: {query}')
#     return tavily.search(query=query)
from langchain_tavily import TavilySearch

llm = ChatOpenAI()
tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({'messages':HumanMessage(content="Give me 3 job search of an AI engineer with Langchain and in bay area on linkedin and list out their details.")})
    print(result)

if __name__ == "__main__":
    main()
