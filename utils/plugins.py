from langchain_community.llms.vllm import VLLM
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_local_llm_and_embeds():
    llm = VLLM(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        tensor_parallel_size=2,
        max_num_seqs=1,
        seed=42,
        dtype="float16",
        gpu_memory_utilization=0.9
    )
    embedding = HuggingFaceEmbeddings(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return llm, embedding