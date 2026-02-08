#!/usr/bin/env python3
"""
Merge ontology nodes by @type field across all files in the directory.
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional


def merge_nodes_by_type(directory: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read all ontology node files and merge them by @type field.

    Args:
        directory: Path to directory containing ontology_node_*.jsonld files

    Returns:
        Dictionary mapping @type to merged node data
    """
    # Dictionary to accumulate data by type
    merged_by_type = defaultdict(lambda: {
        "@type": "",
        "hasCauses": [],
        "hasAssociations": [],
        "hasTemporalCauses": [],
        "hasMediators": [],
        "hasEvidence": []
    })

    # Read all ontology node files
    files = sorted(directory.glob("ontology_node_*.jsonld"))
    print(f"Found {len(files)} ontology node files")

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Process each node in the @graph
            for node in data.get("@graph", []):
                node_type = node.get("@type")
                if not node_type:
                    continue

                # Merge list fields
                for field in ["hasCauses", "hasAssociations", "hasTemporalCauses", "hasMediators"]:
                    value = node.get(field)
                    if value and isinstance(value, list):
                        merged_by_type[node_type][field].extend(value)

                # Merge string fields (hasEvidence)
                evidence = node.get("hasEvidence")
                if evidence and isinstance(evidence, str):
                    merged_by_type[node_type]["hasEvidence"].append(evidence)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # add back @type
    for node_type, node in merged_by_type.items():
        node["@type"] = node_type

    print(f"Merged into {len(merged_by_type)} unique types")
    return merged_by_type


def main():
    # Directory containing ontology node files
    ontology_dir = Path("backpain_data_upgraded_ontology/clbp_abox_ontology")

    if not ontology_dir.exists():
        print(f"Error: Directory {ontology_dir} does not exist")
        return

    # Merge nodes by type
    print("Merging nodes by type...")
    merged = merge_nodes_by_type(ontology_dir)

    # Create output as a single JSON-LD file with @graph as a list
    output = {
        "@graph": list(merged.values())
    }

    # Write merged output
    output_file = Path("backpain_data_upgraded_ontology/clbp_abox_ontology_merged.jsonld")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nMerged ontology written to: {output_file}")


if __name__ == "__main__":
    main()
