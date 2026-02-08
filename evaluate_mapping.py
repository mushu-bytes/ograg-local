from ragas import evaluate
from ragas.metrics import ResponseGroundedness
from ragas.callbacks import RagasTracer
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset
from ragas.run_config import RunConfig
from dotenv import load_dotenv
from llama_index.core.prompts.base import PromptTemplate
from pathlib import Path
from llama_index.core.node_parser import SimpleNodeParser
from utils import get_documents
import pandas as pd
import os
import pickle

# PROMPT Set up
ONTOLOGY_JSONLD_DATA_CREATE_TMPL = """
Here is a context definition for an ontology.

Context Definition:

{context_definition}

-----------------

Generate a JSON-LD using the following data and the above context definition for the given ontology.
Use '@graph' object namespace for the data in JSON-LD.
Be comprehensive and make sure to fill all of the data completely WITHOUT leaving the sentence in "...".
If there are multiple subfields enumerated in a 'List' namespace then do not combine them in a single subfield, keep them as separate subfields to disambiguate.
Ensure that you populate all items in the 'List' namespace, do not leave any item.
Do not include any explanations or apologies in your response.
Do not add any other text other than the generated JSON-LD in your response
Generate in Json format. 
----------------------
Data:

{data}
---------------------
You must adhere to this pydantic model when generating JSON:

{schema}

JSON-LD json:
"""  # noqa: E501

ONTOLOGY_JSONLD_DATA_CREATE_PROMPT = PromptTemplate(ONTOLOGY_JSONLD_DATA_CREATE_TMPL, prompt_type="structured_fill")

# Define base paths
BASE_DIR = Path('/home/damon/ograg2')
BACKPAIN_ONT_DIR = BASE_DIR / 'backpain_data_upgraded_ontology'

# Ontology paths
LOW_LEVEL_ONTOLOGY_FILE = BACKPAIN_ONT_DIR / 'ontology' / 'low_level_ontology.jsonld'
LOW_LEVEL_ONTOLOGY_DIR = BACKPAIN_ONT_DIR / 'low_level_ontology_11_8'
LOW_LEVEL_MODELS_FILE = BASE_DIR / 'models' / 'clbp_causal_ontology_models.py'
LOW_LEVEL_OUTPUT_FILE = BASE_DIR / 'ontology_mapping_evaluation' / 'low_level_mapping_11_8.pkl'

HIGH_LEVEL_ONTOLOGY_FILE = BACKPAIN_ONT_DIR / 'ontology' / 'high_level_ontology.jsonld'
HIGH_LEVEL_ONTOLOGY_DIR = BACKPAIN_ONT_DIR / 'high_level_ontology_11_8'
HIGH_LEVEL_MODELS_FILE = BASE_DIR / 'models' / 'high_level_ontology_models.py'
HIGH_LEVEL_OUTPUT_FILE = BASE_DIR / 'ontology_mapping_evaluation' / 'high_level_mapping_11_8.pkl'

FULL_ONTOLOGY_FILE = BACKPAIN_ONT_DIR / 'ontology' / 'full_ontology.jsonld'
FULL_ONTOLOGY_DIR = BACKPAIN_ONT_DIR / 'full_ontology_11_8'
FULL_MODELS_FILE = BASE_DIR / 'models' / 'full_ontology_models.py'
FULL_OUTPUT_FILE = BASE_DIR / 'ontology_mapping_evaluation' / 'full_mapping_11_8.pkl'

LOW_LEVEL_ONTOLOGY_NO_DEFS_FILE = BACKPAIN_ONT_DIR / 'ontology' / 'low_level_ontology_no_defs.jsonld'
LOW_LEVEL_ONTOLOGY_NO_DEFS_DIR = BACKPAIN_ONT_DIR / 'low_level_ontology_no_defs'
LOW_LEVEL_MODELS_NO_DEFS_FILE = BASE_DIR / 'models' / 'clbp_causal_ontology_models.py'
LOW_LEVEL_OUTPUT_NO_DEFS_FILE = BASE_DIR / 'ontology_mapping_evaluation' / 'low_level_mapping_no_defs.pkl'

HIGH_LEVEL_ONTOLOGY_NO_DEFS_FILE = BACKPAIN_ONT_DIR / 'ontology' / 'high_level_ontology_no_defs.jsonld'
HIGH_LEVEL_ONTOLOGY_NO_DEFS_DIR = BACKPAIN_ONT_DIR / 'high_level_ontology_no_defs'
HIGH_LEVEL_MODELS_NO_DEFS_FILE = BASE_DIR / 'models' / 'high_level_ontology_models.py'
HIGH_LEVEL_OUTPUT_NO_DEFS_FILE = BASE_DIR / 'ontology_mapping_evaluation' / 'high_level_mapping_no_defs.pkl'

FULL_ONTOLOGY_NO_DEFS_FILE = BACKPAIN_ONT_DIR / 'ontology' / 'full_ontology_no_defs.jsonld'
FULL_ONTOLOGY_NO_DEFS_DIR = BACKPAIN_ONT_DIR / 'full_ontology_no_defs'
FULL_MODELS_NO_DEFS_FILE = BASE_DIR / 'models' / 'full_ontology_models.py'
FULL_OUTPUT_NO_DEFS_FILE = BASE_DIR / 'ontology_mapping_evaluation' / 'full_mapping_no_defs.pkl'

SCI_SUBCATEGORIES_ONTOLOGY_FILE = BACKPAIN_ONT_DIR / 'ontology' / 'sci_subcategories_ontology.jsonld'
SCI_SUBCATEGORIES_ONTOLOGY_DIR = BACKPAIN_ONT_DIR / 'sci_subcategories_ontology'
SCI_SUBCATEGORIES_MODELS_FILE = BASE_DIR / 'models' / 'sci_subcategories_ontology_models.py'
SCI_SUBCATEGORIES_OUTPUT_FILE = BASE_DIR / 'ontology_mapping_evaluation' / 'sci_subcategories_mapping.pkl'

SCI_CATEGORIES_ONTOLOGY_FILE = BACKPAIN_ONT_DIR / 'ontology' / 'sci_categories_ontology.jsonld'
SCI_CATEGORIES_ONTOLOGY_DIR = BACKPAIN_ONT_DIR / 'sci_categories_ontology'
SCI_CATEGORIES_MODELS_FILE = BASE_DIR / 'models' / 'sci_categories_ontology_models.py'
SCI_CATEGORIES_OUTPUT_FILE = BASE_DIR / 'ontology_mapping_evaluation' / 'sci_categories_mapping.pkl'

SCI_FULL_ONTOLOGY_FILE = BACKPAIN_ONT_DIR / 'ontology' / 'sci_full_ontology.jsonld'
SCI_FULL_ONTOLOGY_DIR = BACKPAIN_ONT_DIR / 'sci_full_ontology'
SCI_FULL_MODELS_FILE = BASE_DIR / 'models' / 'sci_full_ontology_models.py'
SCI_FULL_OUTPUT_FILE = BASE_DIR / 'ontology_mapping_evaluation' / 'sci_full_mapping.pkl'

# Model and data paths
CLBP_FILE_PATHS = "backpain_data_upgraded_ontology/papers/"
SCI_FILE_PATHS = "/home/damon/scire-pdfs/pdfs"

# LLM Setup
load_dotenv()
pd.options.mode.chained_assignment = None  # default='warn'

# need to set max_workers = 1 or tpm burns up really fast; I can probably increase it a bit higher though
config = RunConfig(max_workers=1)

# gpt-4.1 nano is the cheapest model that I can set temp to 0
llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gpt-4o-mini",
        max_retries=3,
        temperature=0,
        request_timeout=60
    )
)

def hydrate_prompt(ontology_context_definition, schema):
    prompt = ONTOLOGY_JSONLD_DATA_CREATE_PROMPT.format(  # type: ignore
        data="", context_definition=ontology_context_definition, schema=schema
    )
    return prompt

def make_ont_node_paths(prefix, num_files):
    paths = []
    for i in range(num_files):
        paths.append(prefix + f"ontology_node_{i}.jsonld")
    return paths

def read_ontology_node(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        print(e)
        return ""

def evaluate_mapping(ontology_context_definition_path):
    # Initialize variables
    with open(ontology_context_definition_path) as f:
        ontology_context_definition = f.read()
    schema = None
    document_paths = None
    ontology_mapped_nodes_paths = None
    output_file = None
    # will always have chunk_size 8192
    node_parser = SimpleNodeParser.from_defaults(chunk_size=8192)

    match str(ontology_context_definition_path):
        case '/home/damon/ograg2/backpain_data_upgraded_ontology/ontology/low_level_ontology.jsonld':
            print(f"MAPPING THE LOW LEVEL ONTOLOGY FILE")
            with open(LOW_LEVEL_MODELS_FILE) as f:
                schema = f.read()

            document_paths = get_documents(CLBP_FILE_PATHS)
            documents = list(map(lambda x: [x.text], node_parser.get_nodes_from_documents(document_paths)))
            ontology_mapped_nodes_paths = make_ont_node_paths(str(LOW_LEVEL_ONTOLOGY_DIR) + '/', len(documents))
            output_file = LOW_LEVEL_OUTPUT_FILE
        case '/home/damon/ograg2/backpain_data_upgraded_ontology/ontology/high_level_ontology.jsonld':
            print("MAPPING THE HIGH LEVEL ONTOLOGY FILE")
            with open(HIGH_LEVEL_MODELS_FILE) as f:
                schema = f.read()

            document_paths = get_documents(CLBP_FILE_PATHS)
            documents = list(map(lambda x: [x.text], node_parser.get_nodes_from_documents(document_paths)))
            ontology_mapped_nodes_paths = make_ont_node_paths(str(HIGH_LEVEL_ONTOLOGY_DIR) + '/', len(documents))
            output_file = HIGH_LEVEL_OUTPUT_FILE
        case '/home/damon/ograg2/backpain_data_upgraded_ontology/ontology/sci_subcategories_ontology.jsonld':
            print("MAPPING THE SCI SUBCATEGORIES ONTOLOGY FILE")
            with open(SCI_SUBCATEGORIES_MODELS_FILE) as f:
                schema = f.read()

            document_paths = get_documents(SCI_FILE_PATHS)
            documents = list(map(lambda x: [x.text], node_parser.get_nodes_from_documents(document_paths)))
            ontology_mapped_nodes_paths = make_ont_node_paths(str(SCI_SUBCATEGORIES_ONTOLOGY_DIR) + '/', len(documents))
            output_file = SCI_SUBCATEGORIES_OUTPUT_FILE
        case '/home/damon/ograg2/backpain_data_upgraded_ontology/ontology/sci_categories_ontology.jsonld':
            print("MAPPING THE SCI CATEGORIES ONTOLOGY FILE")
            with open(SCI_CATEGORIES_MODELS_FILE) as f:
                schema = f.read()

            document_paths = get_documents(SCI_FILE_PATHS)
            documents = list(map(lambda x: [x.text], node_parser.get_nodes_from_documents(document_paths)))
            ontology_mapped_nodes_paths = make_ont_node_paths(str(SCI_CATEGORIES_ONTOLOGY_DIR) + '/', len(documents))
            output_file = SCI_CATEGORIES_OUTPUT_FILE
        case '/home/damon/ograg2/backpain_data_upgraded_ontology/ontology/full_ontology.jsonld':
            print("MAPPING THE FULL ONTOLOGY FILE")
            with open(FULL_MODELS_FILE) as f:
                schema = f.read()

            document_paths = get_documents(CLBP_FILE_PATHS)
            documents = list(map(lambda x: [x.text], node_parser.get_nodes_from_documents(document_paths)))
            ontology_mapped_nodes_paths = make_ont_node_paths(str(FULL_ONTOLOGY_DIR) + '/', len(documents))
            output_file = FULL_OUTPUT_FILE
        case '/home/damon/ograg2/backpain_data_upgraded_ontology/ontology/low_level_ontology_no_defs.jsonld':
            print("MAPPING THE LOW LEVEL ONTOLOGY NO DEFS FILE")
            with open(LOW_LEVEL_MODELS_NO_DEFS_FILE) as f:
                schema = f.read()

            document_paths = get_documents(CLBP_FILE_PATHS)
            documents = list(map(lambda x: [x.text], node_parser.get_nodes_from_documents(document_paths)))
            ontology_mapped_nodes_paths = make_ont_node_paths(str(LOW_LEVEL_ONTOLOGY_NO_DEFS_DIR) + '/', len(documents))
            output_file = LOW_LEVEL_OUTPUT_NO_DEFS_FILE
        case '/home/damon/ograg2/backpain_data_upgraded_ontology/ontology/high_level_ontology_no_defs.jsonld':
            print("MAPPING THE HIGH LEVEL ONTOLOGY NO DEFS FILE")
            with open(HIGH_LEVEL_MODELS_NO_DEFS_FILE) as f:
                schema = f.read()

            document_paths = get_documents(CLBP_FILE_PATHS)
            documents = list(map(lambda x: [x.text], node_parser.get_nodes_from_documents(document_paths)))
            ontology_mapped_nodes_paths = make_ont_node_paths(str(HIGH_LEVEL_ONTOLOGY_NO_DEFS_DIR) + '/', len(documents))
            output_file = HIGH_LEVEL_OUTPUT_NO_DEFS_FILE
        case '/home/damon/ograg2/backpain_data_upgraded_ontology/ontology/full_ontology_no_defs.jsonld':
            print("MAPPING THE FULL ONTOLOGY NO DEFS FILE")
            with open(FULL_MODELS_NO_DEFS_FILE) as f:
                schema = f.read()

            document_paths = get_documents(CLBP_FILE_PATHS)
            documents = list(map(lambda x: [x.text], node_parser.get_nodes_from_documents(document_paths)))
            ontology_mapped_nodes_paths = make_ont_node_paths(str(FULL_ONTOLOGY_NO_DEFS_DIR) + '/', len(documents))
            output_file = FULL_OUTPUT_NO_DEFS_FILE
        case '/home/damon/ograg2/backpain_data_upgraded_ontology/ontology/sci_full_ontology.jsonld':
            print("MAPPING THE SCI FULL ONTOLOGY FILE")
            with open(SCI_FULL_MODELS_FILE) as f:
                schema = f.read()

            document_paths = get_documents(SCI_FILE_PATHS)
            documents = list(map(lambda x: [x.text], node_parser.get_nodes_from_documents(document_paths)))
            ontology_mapped_nodes_paths = make_ont_node_paths(str(SCI_FULL_ONTOLOGY_DIR) + '/', len(documents))
            output_file = SCI_FULL_OUTPUT_FILE
        case _:
            raise ValueError("Not a real thing")

    ontology_mapped_nodes = list(map(lambda x: read_ontology_node(x), ontology_mapped_nodes_paths))
    prompts = [hydrate_prompt(ontology_context_definition, schema) for _ in range(len(documents))]

    print(f"The number of prompts: {len(prompts)}")
    print(f"The number of ont mapped nodes: {len(ontology_mapped_nodes)}")
    print(f"The number of documents: {len(documents)}")

    dataset = EvaluationDataset.from_pandas(pd.DataFrame({
        "user_input": prompts,
        "response": ontology_mapped_nodes,
        "retrieved_contexts": documents
    }))

    tracer = RagasTracer()
    metrics = [
        ResponseGroundedness()
    ]

    results = evaluate(dataset=dataset, metrics=metrics, llm=llm, callbacks=[tracer], run_config=config)
    results_with_traces = {"results": results, "trace": tracer}

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(results_with_traces, f)

    print(f"RAGAS results saved to {output_file}")

def main():
    if not os.path.exists(LOW_LEVEL_OUTPUT_FILE):
        evaluate_mapping(LOW_LEVEL_ONTOLOGY_FILE)
    else:
        print('=' * 80)
        print("Skipping over evaluation of low level ontology mapping, since it already exists")
        print('=' * 80)

    if not os.path.exists(HIGH_LEVEL_OUTPUT_FILE):
        evaluate_mapping(HIGH_LEVEL_ONTOLOGY_FILE)
    else:
        print('=' * 80)
        print("Skipping over evaluation of high level ontology mapping, since it already exists")
        print('=' * 80)

    if not os.path.exists(FULL_OUTPUT_FILE):
        evaluate_mapping(FULL_ONTOLOGY_FILE)
    else:
        print('=' * 80)
        print("Skipping over evaluation of full ontology mapping, since it already exists")
        print('=' * 80)

    if not os.path.exists(LOW_LEVEL_OUTPUT_NO_DEFS_FILE):
        evaluate_mapping(LOW_LEVEL_ONTOLOGY_NO_DEFS_FILE)
    else:
        print('=' * 80)
        print("Skipping over evaluation of low level ontology no defs mapping, since it already exists")
        print('=' * 80)

    if not os.path.exists(HIGH_LEVEL_OUTPUT_NO_DEFS_FILE):
        evaluate_mapping(HIGH_LEVEL_ONTOLOGY_NO_DEFS_FILE)
    else:
        print('=' * 80)
        print("Skipping over evaluation of high level ontology no defs mapping, since it already exists")
        print('=' * 80)

    if not os.path.exists(FULL_OUTPUT_NO_DEFS_FILE):
        evaluate_mapping(FULL_ONTOLOGY_NO_DEFS_FILE)
    else:
        print('=' * 80)
        print("Skipping over evaluation of full ontology no defs mapping, since it already exists")
        print('=' * 80)

    if not os.path.exists(SCI_SUBCATEGORIES_OUTPUT_FILE):
        evaluate_mapping(SCI_SUBCATEGORIES_ONTOLOGY_FILE)
    else:
        print('=' * 80)
        print("Skipping over evaluation of sci subcategories ontology mapping, since it already exists")
        print('=' * 80)

    if not os.path.exists(SCI_CATEGORIES_OUTPUT_FILE):
        evaluate_mapping(SCI_CATEGORIES_ONTOLOGY_FILE)
    else:
        print('=' * 80)
        print("Skipping over evaluation of sci categories ontology mapping, since it already exists")
        print('=' * 80)

    if not os.path.exists(SCI_FULL_OUTPUT_FILE):
        evaluate_mapping(SCI_FULL_ONTOLOGY_FILE)
    else:
        print('=' * 80)
        print("Skipping over evaluation of sci full categories ontology mapping, since it already exists")
        print('=' * 80)

if __name__ == "__main__":
    main()


