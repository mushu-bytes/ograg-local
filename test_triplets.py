from utils import load_llm_and_embeds, get_config
from knowledge_graph import KGGenerator


if __name__ == "__main__":
    config = get_config()
    
    llm, _ = load_llm_and_embeds(config.model, config.embedding_model)

    print("Instantiated the correct LLM (ChatOpenAI)")
    filename = "backpain_data/kg/ontology/ontology_node_31.jsonld"
    triple_generator = KGGenerator(llm=llm)
    triple_generator.generate_triples(filename)

