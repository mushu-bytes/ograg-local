import json
from rdflib import Graph

ttl_path = "../kg_aug_causal_disc_exp/ontology_info/backpain_ontology.ttl"
jsonld_path = "backpain_jsonld.jsonld"

g = Graph()
g.parse(ttl_path, format="turtle")
jsonld_output = g.serialize(format="json-ld")
with open(jsonld_path, 'w', encoding='utf-8') as f:
        f.write(jsonld_output)