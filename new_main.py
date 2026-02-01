import os
from dotenv import load_dotenv
from typing import List

# LangChain Imports
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Internal File Imports (Ensure these files exist in your folder)
from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

# Load your API keys from .env
load_dotenv()

def main():
    # 1. TOOLS: The 'Hands' of the agent. 
    # Tavily lets the LLM search the real-time internet.
    tools = [TavilySearch()]

    # 2. THE BRAIN: GPT-4o with zero temperature for high logical accuracy.
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3. STRUCTURED BRAIN: 
    # This specifically prepares the LLM to output our Pydantic 'AgentResponse'.
    structured_llm = llm.with_structured_output(AgentResponse)

    # 4. CUSTOM PROMPT:
    # We use your custom template but pass an empty string to format_instructions
    # because 'with_structured_output' handles the formatting instructions via the API.
    react_prompt = PromptTemplate(
        template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        partial_variables={"format_instructions": ""}
    )

    # 5. THE AGENT: The Reasoning Unit.
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt
    )

    # 6. THE EXECUTOR: The Runtime Manager.
    # It handles the loop: Think -> Search -> Observe -> Repeat.
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )

    # 7. THE PIPELINE (LCEL):
    # - Run the executor to get raw search results.
    # - Extract the final text from the 'output' key.
    # - Pass that text to the structured LLM to turn it into a Python Object.
    extract_output = RunnableLambda(lambda x: x['output'])
    chain = agent_executor | extract_output | structured_llm

    # 8. EXECUTION:
    print("--- Starting Agent Workflow ---")
    result = chain.invoke({
        "input": "Search for 3 job postings for an AI engineer with LangChain responsibilities in the Bay Area on LinkedIn."
    })

    # The result is now a clean AgentResponse object.
    print("\n--- Final Structured Result ---")
    print(f"Answer: {result.answer}")
    print(f"Sources: {result.sources}")

if __name__ == "__main__":
    main()