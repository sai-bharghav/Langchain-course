from typing import List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from callbacks import AgentCallbackHandler

# 1. Load environment variables (API Keys) from your .env file
load_dotenv()

# 2. Define a Tool using the @tool decorator. 
# This tells LangChain to convert this function's name and docstring into a format the AI understands.
@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters."""
    print(f"Calling the get_text_length tool with {text}")

    # Basic cleanup: remove quotes or newlines if the AI passed them by mistake
    text = text.strip("'\n").strip('"')

    return len(text)

# 3. Helper function to find the right Python function based on the name the AI sent us.
# The AI sends a string (e.g., "get_text_length"), and we need to find our actual function object.
def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f'Tool with name {tool_name} not found')


if __name__ == '__main__':
    print('Hello Langchain Tools (.bind_tools)!')
    
    # 4. Initialize our tool list and the LLM
    tools = [get_text_length]
    
    # temperature=0 makes the AI predictable and literal (perfect for tool calling).
    # callbacks handles the custom printing we defined in callbacks.py.
    llm = ChatOpenAI(temperature=0, model='gpt-4o', callbacks=[AgentCallbackHandler()])

    # 5. "Bind" the tools to the LLM. 
    # This creates a special version of the LLM that is aware of our get_text_length function.
    llm_with_tools = llm.bind_tools(tools)

    # 6. Setup the initial conversation history with the user's question.
    messages = [HumanMessage(content="What is the length of the text: DOG")]

    # 7. THE AGENT LOOP: This keeps running until the AI provides a final text answer.
    while True:
        # Send the current list of messages (history) to the AI
        ai_message = llm_with_tools.invoke(messages)

        # 8. Check if the AI wants to use a tool. 
        # getattr() safely looks for the 'tool_calls' property without crashing if it's missing.
        tool_calls = getattr(ai_message, "tool_calls", None) or []

        if len(tool_calls) > 0:
            # Step A: Save the AI's request to use a tool into our message history.
            # This is critical so the AI "remembers" it asked to use a tool.
            messages.append(ai_message)
            
            # Step B: Execute every tool call the AI requested.
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id") # Unique ID to match the answer to the request

                # Run our Python function with the arguments the AI provided
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                # Step C: Save the tool's output as a ToolMessage.
                # We include the tool_call_id so the AI knows which request this answer belongs to.
                messages.append(
                    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
                )
            
            # Step D: Jump back to the top of the loop. 
            # We send the updated history (Question + Tool Request + Tool Result) back to the AI.
            continue

        # 9. FINAL ANSWER: If no tool calls were requested, it means the AI is giving us its final reply.
        print(ai_message.content)
        break # Exit the loop