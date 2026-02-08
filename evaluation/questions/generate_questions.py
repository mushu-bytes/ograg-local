import pandas as pd
import json

# TODO: Clean this script to remove everything here that is not plausibility

def plausibility_prompt(var1, var2):
    return (
        f"Is it plausible that {var1} may have a causal relationship with {var2}, either directly or through one or more intermediate variables? "
        "Plausibility refers to the extent to which a hypothesized relationship may be biologically, theoretically, or scientifically reasonable, based on existing knowledge. "
    )


def temporality_prompt(var1, var2):
    return (
        f"Is there a causal relationship where a change in {var1} precedes and leads to a change in {var2}, either directly or through one or more intermediate variables? "
        f"That is, does {var1} temporally precede and influence {var2} in a way consistent with a directional causal link? "
    )

def association_prompt(var1, var2):
    return (
        "Association refers to any statistical dependency between two variables, regardless of strength "
        "or directionality. That is, if knowledge of the value of one variable gives information about the "
        f"other, they are said to be associated. Is there an association between {var1} and {var2}? "

    )

# Read the CSV file
df = pd.read_csv('full_cleaned.csv')

# Generate questions
questions = []

for _, row in df.iterrows():
    var1 = row['var1'] 
    var2 = row['var2']
    var1 = row['var1'] if row['var1'] != 'Sleep' else 'Sleep disturbance'
    var2 = row['var2'] if row['var2'] != 'Sleep' else 'Sleep disturbance'
    
    # Generate all three types of questions for each variable pair
    questions.append({"question": plausibility_prompt(var1, var2)})
    # questions.append({"question": temporality_prompt(var1, var2)})
    # questions.append({"question": association_prompt(var1, var2)})

# Save to JSON file
with open('generated_questions_non_binary.json', 'w') as f:
    json.dump(questions, f, indent=2)

print(f"Generated {len(questions)} questions from {len(df)} variable pairs")