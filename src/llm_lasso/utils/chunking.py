"""
This script implementes chunking for the .json file scraped down from Omim.
"""

import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Chunking omim json docs by attaching gene names and other metadata as metadata tags.
def chunk_by_gene(json_file, output_file, chunk_size=1000, chunk_overlap=200):
    """
    Chunk long fields (textSection, clinicalSynopsis, and geneMapData) from JSON objects,
    and include "full_name" in the metadata.
    Args:
        json_file (str): Path to the input JSON file.
        output_file (str): Path to save the chunked JSON output.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []

    # Count total lines in the input file for progress bar
    with open(json_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Process the JSON file line by line with tqdm
    with open(json_file, "r", encoding="utf-8") as f, tqdm(total=total_lines, desc="Processing JSON lines") as pbar:
        for line in f:
            # Parse the JSON object
            entry = json.loads(line)
            gene_name = entry.get("gene_name", "Unknown")
            preferred_title = entry.get("preferred_title", "Unknown Full Name")
            text_section = entry.get("text_description", "")
            clinical_synopsis = entry.get("clinical_synopsis", "")
            gene_map_data = entry.get("gene_map_data", "")

            # Chunk the text section
            if text_section:
                text_chunks = text_splitter.split_text(text_section)
                for chunk in text_chunks:
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "gene_name": gene_name,
                            "full_name": preferred_title,
                            "section": "text_description"
                        }
                    })

            # Chunk the clinical synopsis
            if clinical_synopsis:
                clinical_chunks = text_splitter.split_text(clinical_synopsis)
                for chunk in clinical_chunks:
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "gene_name": gene_name,
                            "full_name": preferred_title,
                            "section": "clinical_synopsis"
                        }
                    })

            # Chunk the gene map data
            if gene_map_data:
                gene_map_chunks = text_splitter.split_text(gene_map_data)
                for chunk in gene_map_chunks:
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            "gene_name": gene_name,
                            "full_name": preferred_title,
                            "section": "gene_map_data"
                        }
                    })

            # Update progress bar
            pbar.update(1)

    # Save the chunks to the output file
    with open(output_file, "w", encoding="utf-8") as f_out:
        for chunk in chunks:
            f_out.write(json.dumps(chunk) + "\n")

    print(f"Chunked data saved to {output_file}")