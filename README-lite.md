# OG-RAG Lite

This README describes the modifications required for OG-RAG to run on a separate endpoint outside of Azure, Openai, etc.

## Modifications


## Directory Structure
- evaluation:
    - answers: old OG-RAG predictions
    - answers2: contains raw OG-RAG predictions
    - predictions: contains processed OG-RAG predictions (required for running RAGAS)
        - for processing answers into predictions, run the following:
            - python compile_answers.py --answers answers2/low_level_ontology.json --output predictions/low_level_ontology.csv
            - python compile_answers.py --answers answers2/high_level_ontology.json --output predictions/high_level_ontology.csv