# CAI Submission

In this README, the main changes added to the upstream repository for the submission to IEEE CAI are listed.

## Configs:

backpain_configs/ contains all configs necessary to run all experimental controls (CLBP, SCI, with and without descriptions, etc).

## Descriptions:

backpain_data_upgraded_ontology/ontological_definitions.json contains descriptions for each variable used for the CLBP dataset.

## Ontologies:

backpain_data_upgraded_ontology/ontology contains all ontologies used for all experimental controls.

## Pydantic Models:

models/ contains pydantic models used to constrain the local llms to generate structured output, using outlines.

## Evaluation

evaluation/ contains RAGAS evaluation scripts and datasets, such as the SCI and CLBP questions.

