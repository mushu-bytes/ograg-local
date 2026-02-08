from llama_index.core.prompts import PromptTemplate, PromptType
from typing import Any, DefaultDict, Dict, List, Optional
from llama_index.core.retrievers import BaseRetriever

from llama_index.core.retrievers import VectorIndexRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore
from models.reasoning_models import Answer

MAX_TOKENS = 1024

QUERY_PROMPT = """Given the context below, answer the following question.
---------------------
Context:\n {context}\n\n
---------------------
Question: {query_str}
---------------------
Your answer must be in valid JSON, adhering to the following Pydantic schema: 
{schema}
Answer: """

with open('/home/damon/ograg2/models/reasoning_models.py', 'r') as f:
    schema = f.read()

class RAGQueryEngineWithReasoning:
    def __init__(
        self,
        llm: BaseLanguageModel,
        vector_retriever: VectorIndexRetriever,
        **kwargs: Any,
    ):
        self._vector_retriever = vector_retriever
        self.llm = llm
        import outlines
        import openai
        from outlines import Generator

        client = openai.OpenAI(
            base_url="http://localhost:8000/v1",  # Custom endpoint
            api_key="PLACEHOLDER"
        )

        model = outlines.from_vllm(client, "mistralai/Mistral-7B-Instruct-v0.3")
        self.structured_llm = Generator(model, Answer)

    def _retrieve_nodes(self, query_str: str) -> List[NodeWithScore]:
        """Get nodes for Vector Embeddings response and concatenate with KG triples"""
        retrieved_nodes = self._vector_retriever.retrieve(query_str)
        return retrieved_nodes
    

    def query(self, query_str: str, return_context=False, rules=[], **kwargs):
        retrieved_nodes: List[NodeWithScore] = self._retrieve_nodes(query_str)
        retrieved_nodes_text = "\n".join([node.text for node in retrieved_nodes])

        response = self.structured_llm(QUERY_PROMPT.format(
                                            context=retrieved_nodes_text + "\n".join(rules),
                                            query_str=query_str,
                                            schema=schema
                                        ), max_tokens=MAX_TOKENS)
        response = Answer.model_validate_json(response)
        if return_context:
            return response, retrieved_nodes_text
        return response 

