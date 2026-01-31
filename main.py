from dotenv import load_dotenv
# We use 'langchain_classic' for standard ReAct patterns that are stable and well-documented.
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# --- CONCEPT: THE REASONING ENGINE vs. THE EXECUTION BODY ---
# Think of the 'Agent' as the brain: it plans what to do next.
# Think of the 'AgentExecutor' as the body: it actually carries out those plans.
# -----------------------------------------------------------

load_dotenv()

# 1. TOOLS: These are the "hands" of the agent. 
# Without tools, an LLM can only 'talk' from its memory. 
# TavilySearch gives it the ability to 'act' by searching the live internet.
tools = [TavilySearch()]

# 2. THE BRAIN: We use GPT-4o with temperature=0.
# We set temperature to 0 because we want the agent to be focused and factual, 
# not 'creative' or 'hallucinatory' when following search steps.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. THE HUB & PROMPT: The "Instructions Manual".
# 'hub.pull' downloads a famous prompt template (hwchase17/react).
# This template tells the LLM: "Think step-by-step using: Thought, Action, Action Input, Observation."
react_prompt = hub.pull("hwchase17/react")

# 4. THE AGENT: The "Decision Maker".
# 'create_react_agent' assembles the logic. It combines the Brain (LLM), 
# the Manual (Prompt), and the Hands (Tools) into a single planning unit.
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

# 5. THE EXECUTOR: The "Manager".
# This is the most critical part. It handles the loop: 
# It runs the agent's plan, executes the tool, feeds the result back, and repeats.
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,               # Shows the 'internal monologue' in the terminal.
    handle_parsing_errors=True, # Safety: If the LLM makes a typo, the manager asks it to try again.
    max_iterations=5            # Security: Stops the agent if it gets stuck in an infinite search loop.
)

# We define the 'chain' as our executor so we can call it later.
chain = agent_executor

def main():
    # 6. INVOCATION: Starting the engine.
    # We pass the user's input into the 'input' key that the ReAct prompt expects.
    result = chain.invoke(
        input={
            "input": "Search for 3 job postings for an AI engineer where langchain is the job responsibility where location is in the bay area on linkedin and list their details.",
        }
    )
    # The 'result' will contain the final answer after the agent has finished its work.
    print(result)

if __name__ == "__main__":
    main()