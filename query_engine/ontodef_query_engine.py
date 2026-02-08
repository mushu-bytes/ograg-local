from typing import Any, DefaultDict, Dict, List, Optional, Union, Set
from llama_index.core.retrievers import BaseRetriever

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
from llama_index.core.node_parser import SimpleNodeParser
from models.reasoning_models import Answer
from utils import load_graph_nodes, load_graph_nodes_chunks, cosine_similarity, flatten_tree, load_ont_nodes
import logging
import json

MAX_TOKENS = 1024

with open('/home/damon/ograg2/models/reasoning_models.py', 'r') as f:
    schema = f.read()

with open('/home/damon/ograg2/backpain_data_upgraded_ontology/ontological_definitions.json', 'r') as f:
    definitions = json.load(f)

RAG_QUERY_PROMPT = """Given the context below, answer the following question. 

Context: {context}

Question: {query_str}

Your answer must be in valid JSON, adhering to the following Pydantic schema: 
{schema}

Answer:
"""


class OntoDefQueryEngine:
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
        relevant_context = self.retrieve_context(query_str=query_str)
        try:
            response = self.structured_llm(
                RAG_QUERY_PROMPT.format(
                    context=relevant_context, 
                    query_str=query_str,
                    schema=schema
                ),
                max_tokens=MAX_TOKENS
            )
            # in ontological definitions, you have to convert Sleep to Sleep disturbance
            response = Answer.model_validate_json(response)
        except Exception as e:
            logging.error(f"Failed to validate response: {e}")
            logging.error(f"Raw query_str: {query_str}")
            # Return default Answer object
            response = Answer(conclusion=False, reasoning="Failed to parse response")
        if return_context:
            return response, relevant_context
        return response

    def retrieve_context(self, query_str: str):
        contexts = ["\n"]
        for key, value in definitions.items():
            if query_str.find(key) == -1:
                continue
            contexts.append(value)
            if len(contexts) == 3:
                break
        return "\n".join(contexts)