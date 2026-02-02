from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def multiply(x:float,y:float)->float:
    """Multiplies two numbers."""
    return x*y

if __name__=='__main__':
    print('Hello Tool calling')
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system','You are a helpful assistant that can use tools.'),
            ('human','{input}'),
            ('placeholder','{agent_scratchpad}')
        ]
    )
    tools = [multiply,TavilySearch()]
    llm = ChatOpenAI(temperature=0,model='gpt-4o')
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )  
    agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)
    response = agent_executor.invoke({'input':'What is weather in Hyderabad in India? and how different is it from the weather in Irving Texas? Please keep the answer in Celsius'})
    print(response)

