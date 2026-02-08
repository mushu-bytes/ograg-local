#!/usr/bin/env python3
"""
Quick script to check how many nodes would be created from a directory of files.
"""
import sys
from pathlib import Path

from llama_index.core.node_parser import SimpleNodeParser
from utils.utils import get_documents


def count_nodes_in_directory(
    input_dir: str,
    chunk_size: int = 1000,
    smart_pdf: bool = True,
    subdir: bool = False
) -> None:
    """
    Count how many nodes would be created from files in a directory.

    Args:
        input_dir: Path to directory containing files
        chunk_size: Size of text chunks for node parsing
        smart_pdf: Whether to use smart PDF loader
        subdir: Whether to process subdirectories
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    print(f"Chunk Size: {chunk_size}")
    # Load documents using the same function as ontology_mapping
    print(f"\nLoading documents from {input_dir} (smart_pdf={smart_pdf}, subdir={subdir})...")
    documents = get_documents(str(input_dir), subdir=subdir, smart_pdf=smart_pdf, full_text=True)
    print(f"Loaded {len(documents)} documents")

    # Parse into nodes
    print(f"\nParsing documents into nodes (chunk_size={chunk_size})...")
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
    nodes = node_parser.get_nodes_from_documents(documents)

    print(f"\n{'='*60}")
    print(f"TOTAL NODES: {len(nodes)}")
    print(f"{'='*60}")

    # Show node size distribution
    if nodes:
        node_sizes = [len(node.text) for node in nodes]
        print(f"\nNode size statistics:")
        print(f"  Min: {min(node_sizes)} characters")
        print(f"  Max: {max(node_sizes)} characters")
        print(f"  Avg: {sum(node_sizes) / len(node_sizes):.0f} characters")

        # Show first few nodes
        print(f"\nFirst 3 nodes (preview):")
        for i, node in enumerate(nodes[:3]):
            preview = node.text[:200].replace('\n', ' ')
            print(f"\n  Node {i}:")
            print(f"    Size: {len(node.text)} chars")
            print(f"    Preview: {preview}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_ontology_nodes.py <input_directory> [chunk_size] [smart_pdf] [subdir]")
        print("\nExample:")
        print("  python count_ontology_nodes.py backpain_data_upgraded_ontology/pdf/")
        print("  python count_ontology_nodes.py backpain_data_upgraded_ontology/pdf/ 4096")
        print("  python count_ontology_nodes.py backpain_data_upgraded_ontology/pdf/ 8192 True False")
        sys.exit(1)

    input_dir = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    smart_pdf = sys.argv[3].lower() != 'false' if len(sys.argv) > 3 else True
    subdir = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False

    count_nodes_in_directory(input_dir, chunk_size, smart_pdf, subdir)
