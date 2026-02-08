from pydantic import BaseModel, Field
from typing import List

class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: bool = Field(..., description="The culminating final conclusion or answer to the question")