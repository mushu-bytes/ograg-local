import json
import pandas as pd
import argparse
from pathlib import Path

def compile_predictions(answers_path, csv_path):
    """
    Compile predictions from answers.json and match them to CSV rows.
    Each 3 consecutive objects in answers.json correspond to one row in full_cleaned.csv.
    """
    
    # Load the answers JSON file
    with open(answers_path, 'r') as f:
        answers = json.load(f)
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize result list
    results = []
    
    # Process answers in groups of 3
    for i in range(0, len(answers), 3):
        if i + 2 < len(answers):  # Ensure we have 3 predictions
            row_index = i // 3
            if row_index < len(df):
                row = df.iloc[row_index]
                
                # Extract the 3 predictions for this row
                pred1 = answers[i]["conclusion"]
                pred2 = answers[i + 1]["conclusion"] 
                pred3 = answers[i + 2]["conclusion"]

                # Extract the 3 reasonings for this row
                reasoning1 = answers[i]["reasoning"]
                reasoning2 = answers[i + 1]["reasoning"] 
                reasoning3 = answers[i + 2]["reasoning"]
                
                # Compile result
                result = {
                    "Var1": row["var1"],
                    "Var2": row["var2"],
                    "Plausibility": pred1,
                    "Temporality": pred2,
                    "Association": pred3,
                    "Plausibility Reasoning": "\n".join(reasoning1),
                    "Temporality Reasoning": "\n".join(reasoning2),
                    "Association Reasoning": "\n".join(reasoning3),
                    "Plausibility Query": answers[i]["question"],
                    "Temporality Query": answers[i + 1]["question"],
                    "Association Query": answers[i + 2]["question"],
                    "Chunks": answers[i]["retrieved_context"]
                }
                
                results.append(result)
    
    return results

def save_compiled_results(results, output_path="compiled_predictions.csv"):
    """Save compiled results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} compiled predictions to {output_path}")

def analyze_predictions(results):
    """Basic analysis of the predictions"""
    total_rows = len(results)
    
    # Count agreement between predictions
    all_agree = sum(1 for r in results if r["Plausibility"] == r["Association"] == r["Temporality"])
    two_agree = sum(1 for r in results if (r["Plausibility"] == r["Association"]) or 
                   (r["Plausibility"] == r["Temporality"]) or 
                   (r["Association"] == r["Temporality"]))
    
    print(f"Total rows processed: {total_rows}")
    print(f"All 3 predictions agree: {all_agree} ({all_agree/total_rows*100:.1f}%)")
    print(f"At least 2 predictions agree: {two_agree} ({two_agree/total_rows*100:.1f}%)")
    
    return {
        "total_rows": total_rows,
        "all_agree": all_agree,
        "two_agree": two_agree
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Compile predictions from answers JSON and match to CSV rows")
    parser.add_argument("--answers", "-a", 
                       default="answers/onto_def_answers.json",
                       help="Path to the answers JSON file (default: answers/onto_def_answers.json)")
    parser.add_argument("--csv", "-c",
                       default="questions/full_cleaned.csv", 
                       help="Path to the CSV file (default: questions/full_cleaned.csv)")
    parser.add_argument("--output", "-o",
                       default="compiled_predictions.csv",
                       help="Output CSV file path (default: compiled_predictions.csv)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"Compiling predictions from {args.answers}...")
    
    # Compile the predictions
    results = compile_predictions(args.answers, args.csv)
    
    # Save results
    save_compiled_results(results, args.output)
    print(results[0])
    
    # Basic analysis
    stats = analyze_predictions(results)
    
    # Display first few results as example
    print("\nFirst 3 compiled results:")
    for i, result in enumerate(results[:3]):
        print(f"\n{result['Var1']} -> {result['Var2']}")
        print(f"Predictions: {result['Plausibility']}, {result['Association']}, {result['Temporality']}")