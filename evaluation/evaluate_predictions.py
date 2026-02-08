from ragas import evaluate
from ragas.metrics import ContextRelevance, ResponseGroundedness
from ragas.callbacks import RagasTracer
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset
from ragas.run_config import RunConfig
from dotenv import load_dotenv
import pandas as pd
import json
import pickle
import os

load_dotenv()
pd.options.mode.chained_assignment = None  # default='warn'

# SCI Configuration
SCI_RAG_FILE = "answers2/sci_rag.json"
SCI_CATEGORIES_FILE = "answers2/sci_categories_ontology.json"
SCI_SUBCATEGORIES_FILE = "answers2/sci_subcategories_ontology.json"
SCI_FULL_FILE = "answers2/sci_full_ontology.json"
SCI_RAG_OUTPUT = "ragas_eval/sci_rag_eval.pkl"
SCI_CATEGORIES_OUTPUT = "ragas_eval/sci_categories_eval.pkl"
SCI_SUBCATEGORIES_OUTPUT = "ragas_eval/sci_subcategories_eval.pkl"
SCI_FULL_OUTPUT = "ragas_eval/sci_full_eval.pkl"

# CLBP Configuration
CLBP_RAG_FILE = "answers2/rag_11_8.json"
CLBP_HIGH_LEVEL_FILE = "answers2/high_level_ontology_11_8.json"
CLBP_LOW_LEVEL_FILE = "answers2/low_level_ontology_11_8.json"
CLBP_FULL_FILE = "answers2/full_ontology_11_8.json"
CLBP_RAG_OUTPUT = "ragas_eval/rag_eval_11_8.pkl"
CLBP_HIGH_LEVEL_OUTPUT = "ragas_eval/high_level_ontology_eval_11_8.pkl"
CLBP_LOW_LEVEL_OUTPUT = "ragas_eval/low_level_ontology_eval_11_8.pkl"
CLBP_FULL_OUTPUT = "ragas_eval/full_ontology_eval_11_8.pkl"

# CLBP Configuration (no defs)
CLBP_HIGH_LEVEL_FILE_NO_DEFS = "answers2/high_level_ontology_no_defs.json"
CLBP_LOW_LEVEL_FILE_NO_DEFS = "answers2/low_level_ontology_no_defs.json"
CLBP_FULL_FILE_NO_DEFS = "answers2/full_ontology_no_defs.json"
CLBP_HIGH_LEVEL_OUTPUT_NO_DEFS = "ragas_eval/high_level_ontology_eval_no_defs.pkl"
CLBP_LOW_LEVEL_OUTPUT_NO_DEFS = "ragas_eval/low_level_ontology_eval_no_defs.pkl"
CLBP_FULL_OUTPUT_NO_DEFS = "ragas_eval/full_ontology_eval_no_defs.pkl"

# need to set max_workers = 1 or tpm burns up really fast
config = RunConfig(max_workers=1)

# gpt-4o-mini is the cheapest model that I can set temp to 0
llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="gpt-4o-mini",
        max_retries=3,
        temperature=0,
        request_timeout=60
    )
)

def load_and_prepare_data(json_file: str) -> pd.DataFrame:
    """
    Load the JSON file and prepare it in RAGAS format.

    Args:
        json_file: Path to the JSON file containing questions, answers, and contexts

    Returns:
        DataFrame with columns: user_input, response, retrieved_contexts
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Prepare data in RAGAS format
    prepared_data = []
    for item in data:
        # Convert retrieved_context list of dicts to list of strings
        contexts = [str(d) for d in item.get("retrieved_context", [])]

        prepared_data.append({
            "user_input": item.get("question", ""),
            "response": item.get("answer", ""),
            "retrieved_contexts": contexts
        })

    return pd.DataFrame(prepared_data)

def evaluate_ontology_rag(answers_file: str, output_file: str):
    """
    Evaluate ontology answers using RAGAS metrics:
    - ContextRelevancy: Measures how relevant the retrieved contexts are to the question
    - ResponseGroundedness: Measures how well the response is grounded in the retrieved contexts

    Args:
        answers_file: Path to the JSON file containing answers
        output_file: Path to save the evaluation results
    """
    print(f"Loading data from {answers_file}...")
    df = load_and_prepare_data(answers_file)

    print(f"Loaded {len(df)} question-answer pairs")
    print(f"\nSample data:")
    print(df.head(2))

    # Create RAGAS dataset
    dataset = EvaluationDataset.from_pandas(df)

    # Initialize tracer and metrics
    tracer = RagasTracer()
    metrics = [
        ContextRelevance(),
        ResponseGroundedness()
    ]

    print(f"\nRunning RAGAS evaluation with metrics: {[m.__class__.__name__ for m in metrics]}")
    print("This may take a while...")

    # Run evaluation
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        callbacks=[tracer],
        run_config=config
    )

    # Package results with trace
    results_with_traces = {
        "results": results,
        "trace": tracer,
    }

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(results_with_traces, f)

    print(f"\nEvaluation complete!")
    print(f"Results saved to {output_file}")

    return results_with_traces

if __name__ == "__main__":
    # Evaluate SCI Categories
    if os.path.exists(SCI_CATEGORIES_OUTPUT):
        print("=" * 80)
        print(f"Results already exist at {SCI_CATEGORIES_OUTPUT}")
        print("Skipping SCI categories evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating SCI Categories Ontology")
        print("=" * 80)
        evaluate_ontology_rag(SCI_CATEGORIES_FILE, SCI_CATEGORIES_OUTPUT)

    print("\n")

    # Evaluate SCI Subcategories
    if os.path.exists(SCI_SUBCATEGORIES_OUTPUT):
        print("=" * 80)
        print(f"Results already exist at {SCI_SUBCATEGORIES_OUTPUT}")
        print("Skipping SCI subcategories evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating SCI Subcategories Ontology")
        print("=" * 80)
        evaluate_ontology_rag(SCI_SUBCATEGORIES_FILE, SCI_SUBCATEGORIES_OUTPUT)

    print("\n")

    # Evaluate SCI Full
    if os.path.exists(SCI_FULL_OUTPUT):
        print("=" * 80)
        print(f"Results already exist at {SCI_FULL_OUTPUT}")
        print("Skipping SCI full ontology evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating SCI Full Ontology")
        print("=" * 80)
        evaluate_ontology_rag(SCI_FULL_FILE, SCI_FULL_OUTPUT)

    print("\n")

    # Evaluate SCI RAG
    if os.path.exists(SCI_RAG_OUTPUT):
        print("=" * 80)
        print(f"Results already exist at {SCI_RAG_OUTPUT}")
        print("Skipping SCI rag evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating SCI RAG")
        print("=" * 80)
        evaluate_ontology_rag(SCI_RAG_FILE, SCI_RAG_OUTPUT)

    print("\n")

    # Evaluate cLBP High Level
    if os.path.exists(CLBP_HIGH_LEVEL_OUTPUT):
        print("=" * 80)
        print(f"Results already exist at {CLBP_HIGH_LEVEL_OUTPUT}")
        print("Skipping cLBP high level ontology evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating cLBP High Level Ontology (11_8)")
        print("=" * 80)
        evaluate_ontology_rag(CLBP_HIGH_LEVEL_FILE, CLBP_HIGH_LEVEL_OUTPUT)

    print("\n")

    # Evaluate cLBP Low Level
    if os.path.exists(CLBP_LOW_LEVEL_OUTPUT):
        print("=" * 80)
        print(f"Results already exist at {CLBP_LOW_LEVEL_OUTPUT}")
        print("Skipping cLBP low level ontology evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating cLBP Low Level Ontology (11_8)")
        print("=" * 80)
        evaluate_ontology_rag(CLBP_LOW_LEVEL_FILE, CLBP_LOW_LEVEL_OUTPUT)

    print("\n")

    # Evaluate cLBP Full
    if os.path.exists(CLBP_FULL_OUTPUT):
        print("=" * 80)
        print(f"Results already exist at {CLBP_FULL_OUTPUT}")
        print("Skipping cLBP full ontology evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating cLBP Full Ontology (11_8)")
        print("=" * 80)
        evaluate_ontology_rag(CLBP_FULL_FILE, CLBP_FULL_OUTPUT)

    print("\n")

    # Evaluate cLBP RAG
    if os.path.exists(CLBP_RAG_OUTPUT):
        print("=" * 80)
        print(f"Results already exist at {CLBP_RAG_OUTPUT}")
        print("Skipping cLBP RAG evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating cLBP RAG")
        print("=" * 80)
        evaluate_ontology_rag(CLBP_RAG_FILE, CLBP_RAG_OUTPUT)

    # NO DEFS EVALUATION OF CLBP
    # Evaluate cLBP High Level
    if os.path.exists(CLBP_HIGH_LEVEL_OUTPUT_NO_DEFS):
        print("=" * 80)
        print(f"Results already exist at {CLBP_HIGH_LEVEL_OUTPUT_NO_DEFS}")
        print("Skipping cLBP high level ontology evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating cLBP High Level Ontology no defs")
        print("=" * 80)
        evaluate_ontology_rag(CLBP_HIGH_LEVEL_FILE_NO_DEFS, CLBP_HIGH_LEVEL_OUTPUT_NO_DEFS)

    print("\n")

    # Evaluate cLBP Low Level
    if os.path.exists(CLBP_LOW_LEVEL_OUTPUT_NO_DEFS):
        print("=" * 80)
        print(f"Results already exist at {CLBP_LOW_LEVEL_OUTPUT_NO_DEFS}")
        print("Skipping cLBP low level ontology evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating cLBP Low Level Ontology no defs")
        print("=" * 80)
        evaluate_ontology_rag(CLBP_LOW_LEVEL_FILE_NO_DEFS, CLBP_LOW_LEVEL_OUTPUT_NO_DEFS)

    print("\n")

    # Evaluate cLBP Full
    if os.path.exists(CLBP_FULL_OUTPUT_NO_DEFS):
        print("=" * 80)
        print(f"Results already exist at {CLBP_FULL_OUTPUT_NO_DEFS}")
        print("Skipping cLBP full ontology evaluation...")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Evaluating cLBP Full Ontology no defs")
        print("=" * 80)
        evaluate_ontology_rag(CLBP_FULL_FILE_NO_DEFS, CLBP_FULL_OUTPUT_NO_DEFS)

    print("\n")