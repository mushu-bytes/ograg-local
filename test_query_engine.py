from query_engine import OntoHyperGraphQueryEngine
from utils import load_llm_and_embeds, get_config

config = get_config()
llm, embeddings = load_llm_and_embeds(config.model, config.embedding_model)
query_engine = OntoHyperGraphQueryEngine.from_ontology_path(
                    ontology_nodes_path=f'{config.data.kg_storage_path}',
                    llm=llm,
                    embed_model=embeddings,
                    vector_retriever=None
                )

query_str = "Is it plausible that Age may have a causal relationship with Alcohol, either directly or through one or more intermediate variables? Plausibility refers to the extent to which a hypothesized relationship may be biologically, theoretically, or scientifically reasonable, based on existing knowledge. Output TRUE if it is plausible, FALSE if not."

nodes, context = query_engine.retrieve_context(query_str=query_str)
print(f"Number of nodes retrieved: {len(nodes)}")
for node in nodes:
    print(f"Key: {node.key}, Value: {node.value}")
