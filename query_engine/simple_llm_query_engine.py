from typing import Any, DefaultDict, Dict, List, Optional, Union, Set
from langchain_core.language_models import BaseLanguageModel
import logging
from models.reasoning_models import Answer

MAX_TOKENS = 1024

with open('/home/damon/ograg2/models/reasoning_models.py', 'r') as f:
    schema = f.read()

SIMPLE_QUERY_PROMPT = """Answer the following question directly without any additional context.

Question: {query_str}

Your answer must be in valid JSON, adhering to the following Pydantic schema: 
{schema}

Answer:
"""


class SimpleLLMQueryEngine:
    def __init__(self, llm: BaseLanguageModel):
        self._llm = llm

        import outlines
        import openai
        from outlines import Generator
        from models.reasoning_models import Answer

        client = openai.OpenAI(
            base_url="http://localhost:8000/v1",  # Custom endpoint
            api_key="PLACEHOLDER"
        )

        model = outlines.from_vllm(client, "mistralai/Mistral-7B-Instruct-v0.3")
        self.structured_llm = Generator(model, Answer)

    def query(self, query_str: str, top_k=5, context_length: int=1024, return_context: bool=False, rules=[], **kwargs):
        try:
            response = self.structured_llm(
                SIMPLE_QUERY_PROMPT.format(
                    query_str=query_str,
                    schema=schema
                ),
                max_tokens=MAX_TOKENS
            )
            response = Answer.model_validate_json(response)
        except Exception as e:
            logging.error(f"Failed to validate response: {e}")
            logging.error(f"Raw query_str: {query_str}")
            # Return default Answer object
            response = Answer(conclusion=False, reasoning="Failed to parse response")
        
        if return_context:
            return response, ""  # No context for simple LLM
        return response