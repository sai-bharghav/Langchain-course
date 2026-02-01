from typing import List

from pydantic import BaseModel, Field

# --- CONCEPT: STRUCTURED OUTPUT ---
# These classes act as 'Blueprints' or 'Templates'.
# They tell the AI: "You cannot just give me a random paragraph; 
# you must fill out these specific fields correctly."
# ----------------------------------

class Source(BaseModel):
    """Schema for a source used by the agent"""

    # Field descriptions are important because the LLM reads them 
    # to understand what data to put into the variable.
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    # The main text response for the user.
    answer: str = Field(description="The agent's answer to the query")

    # A collection of Source objects defined above.
    sources:List[Source]=Field(
        default_factory=list,# If no sources are found, it provides an empty list instead of an error.
        description="List of sources used by the agent to formulate the answer"
    )