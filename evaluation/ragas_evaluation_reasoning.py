from ragas import evaluate
from ragas.metrics import ResponseGroundedness, ContextRelevance
from ragas.callbacks import RagasTracer
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from typing import Dict
from ragas import EvaluationDataset
from typing import List
from ragas.run_config import RunConfig
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
pd.options.mode.chained_assignment = None  # default='warn'

def load_and_clean_predictions_ontodefs(prediction_file: str, edge_list_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and clean prediction data by standardizing variable names and merging with edge list.
    
    Args:
        prediction_file: Path to the prediction CSV file
        edge_list_df: DataFrame containing expert edges
        
    Returns:
        Cleaned DataFrame with Label column indicating true edges
    """
    # Load predictions
    df = pd.read_csv(prediction_file)
    
    # Standardize variable names (Sleep -> Sleep disturbance)
    df["Var1"] = df["Var1"].replace("Sleep", "Sleep disturbance")
    df["Var2"] = df["Var2"].replace("Sleep", "Sleep disturbance")
    
    # Convert Chunks to list format for RAGAS compatibility
    df["Chunks"] = df["Chunks"].apply(lambda x: [x])
    
    # Merge with edge list to create labels
    df = pd.merge(df, edge_list_df, how='outer', on=["Var1", "Var2"], indicator=True)
    df["Label"] = df["_merge"] == "both"
    df = df.drop(columns=["_merge"])
    
    return df

def load_and_clean_predictions(prediction_file: str, edge_list_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and clean prediction data by standardizing variable names and merging with edge list.
    
    Args:
        prediction_file: Path to the prediction CSV file
        edge_list_df: DataFrame containing expert edges
        
    Returns:
        Cleaned DataFrame with Label column indicating true edges
    """
    import ast
    
    # Load predictions
    df = pd.read_csv(prediction_file)
    
    # Standardize variable names (Sleep -> Sleep disturbance)
    df["Var1"] = df["Var1"].replace("Sleep", "Sleep disturbance")
    df["Var2"] = df["Var2"].replace("Sleep", "Sleep disturbance")
    
    # Convert Chunks using ast.literal_eval to get list of dictionaries, then convert each dict to string
    def process_chunks(x):
        try:
            # Use ast.literal_eval to parse the string representation of list of dictionaries
            dict_list = ast.literal_eval(x)
            # Convert each dictionary to a string
            return [str(d) for d in dict_list]
        except Exception as e:
            return x.split('\n')
    
    df["Chunks"] = df["Chunks"].apply(process_chunks)
    
    # Merge with edge list to create labels
    df = pd.merge(df, edge_list_df, how='outer', on=["Var1", "Var2"], indicator=True)
    df["Label"] = df["_merge"] == "both"
    df = df.drop(columns=["_merge"])
    
    return df

# Load edge list once
edge_list = pd.read_csv("../data/expert_edges_latest.csv")

# Load and clean both datasets
# ontodef = load_and_clean_predictions_ontodefs("predictions/onto_def_predictions.csv", edge_list)
low_level_ontology = load_and_clean_predictions("predictions/low_level_ontology_11_8.csv", edge_list)
high_level_ontology = load_and_clean_predictions("predictions/high_level_ontology_11_8.csv", edge_list)
full_ontology = load_and_clean_predictions("predictions/full_ontology_11_8.csv", edge_list)
rag = load_and_clean_predictions("predictions/rag.csv", edge_list)

# need to set max_workers = 1 or tpm burns up really fast
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

# convert to RAGAS dataset and eval
def eval(df: pd.DataFrame) -> Dict:
    dataset = EvaluationDataset.from_pandas(df)
    tracer = RagasTracer()
    metrics = [
        ResponseGroundedness(),
        ContextRelevance()
        # LLMContextPrecisionWithoutReference()
    ]

    results = evaluate(dataset=dataset, metrics=metrics, llm=llm, callbacks=[tracer], run_config=config)
    return {"results": results, "trace": tracer}

def eval_pat(df: pd.DataFrame, treatments: List[str] = ["Plausibility"]):
    ragas_columns = ["user_input", "response", "retrieved_contexts"]
    pat = {
        "Plausibility": ["Plausibility Query", "Plausibility Reasoning", "Chunks"],
        "Association": ["Association Query", "Association Reasoning", "Chunks"],
        "Temporality": ["Temporality Query", "Temporality Reasoning", "Chunks"]
    }
    results = {}
    for treatment in treatments: 
        subset = df[pat[treatment]]
        subset.columns = ragas_columns
        results[treatment] = eval(subset)
        
    return results

import pickle
import os

# Check if results already exist, otherwise run evaluations
# ontodef_pkl = 'ragas_eval/ontodef_eval.pkl'
low_level_pkl = 'ragas_eval/low_level_ontology_eval_11_8.pkl'
high_level_pkl = 'ragas_eval/high_level_ontology_eval_11_8.pkl'
full_pkl = 'ragas_eval/full_ontology_eval_11_8.pkl'
rag_pkl = 'ragas_eval/rag_eval.pkl'

# if os.path.exists(ontodef_pkl):
#     print(f"Results already exist at {ontodef_pkl}, skipping evaluation")
# else:
#     print(f"No existing results found. Running evaluation for ontodef...")
#     ontodef_eval = eval_pat(ontodef)
#     os.makedirs('ragas_eval', exist_ok=True)
#     pickle.dump(ontodef_eval, open(ontodef_pkl, 'wb'))
#     print(f"Saved results to {ontodef_pkl}")

if os.path.exists(low_level_pkl):
    print(f"Results already exist at {low_level_pkl}, skipping evaluation")
else:
    print(f"No existing results found. Running evaluation for low_level_ontology...")
    low_level_ontology_eval = eval_pat(low_level_ontology)
    os.makedirs('ragas_eval', exist_ok=True)
    pickle.dump(low_level_ontology_eval, open(low_level_pkl, 'wb'))
    print(f"Saved results to {low_level_pkl}")

if os.path.exists(high_level_pkl):
    print(f"Results already exist at {high_level_pkl}, skipping evaluation")
else:
    print(f"No existing results found. Running evaluation for high_level_ontology...")
    high_level_ontology_eval = eval_pat(high_level_ontology)
    os.makedirs('ragas_eval', exist_ok=True)
    pickle.dump(high_level_ontology_eval, open(high_level_pkl, 'wb'))
    print(f"Saved results to {high_level_pkl}")

if os.path.exists(full_pkl):
    print(f"Results already exist at {full_pkl}, skipping evaluation")
else:
    print(f"No existing results found. Running evaluation for full_ontology...")
    full_ontology_eval = eval_pat(full_ontology)
    os.makedirs('ragas_eval', exist_ok=True)
    pickle.dump(full_ontology_eval, open(full_pkl, 'wb'))
    print(f"Saved results to {full_pkl}")

if os.path.exists(rag_pkl):
    print(f"Results already exist at {rag_pkl}, skipping evaluation")
else:
    print(f"No existing results found. Running evaluation for rag...")
    rag_eval = eval_pat(rag)
    os.makedirs('ragas_eval', exist_ok=True)
    pickle.dump(rag_eval, open(rag_pkl, 'wb'))
    print(f"Saved results to {rag_pkl}")