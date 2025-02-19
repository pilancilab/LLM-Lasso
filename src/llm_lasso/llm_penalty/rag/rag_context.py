"""
This script offers the main function to process retrieved RAG documents using features provided in omim_RAG_process.py and pubMed_RAG_process.py.
"""

from llm_lasso.llm_penalty.rag.omim_RAG_process import *
from llm_lasso.llm_penalty.rag.pubMed_RAG_process import pubmed_retrieval
from llm_lasso.utils.score_collection import retrieval_docs, get_unique_docs
from langchain_community.vectorstores import Chroma


def get_rag_context(
    batch_genes: list[str],
    category: str,
    vectorstore: Chroma,
    model: LLMQueryWrapperWithMemory,
    omim_api_key: str,
    pubmed_docs=False,
    filtered_cancer_docs = False,
    summarized_gene_docs = False,
    original_docs = True,
    default_num_docs = 3,
    small = False
):
    """
    Retrieve RAG context for gene data, combining three different RAG processes.
    """
    context = ""
    skip_genes = set()
    if pubmed_docs:
        print("Retrieving pubmed")
        context += pubmed_retrieval(batch_genes, category, model) + "\n"

    if filtered_cancer_docs:
        print("Retrieving cancer docs")
        (add_ctx, skip_genes) = get_filtered_cancer_docs_and_genes_found(
            batch_genes, vectorstore.as_retriever(search_kwargs={"k": 100}),
            model, category
        )
        context += add_ctx + "\n"

    if summarized_gene_docs:
        print("Retrieving gene docs")
        preamble = "\nAdditional gene information: \n" if context.strip() != "" else ""
        context += preamble + get_summarized_gene_docs(
            [gene for gene in batch_genes if gene not in skip_genes],
            model, omim_api_key
        ) + "\n"

    if original_docs:
        print("Retrieving original docs")
        docs = retrieval_docs(
            batch_genes, category,
            vectorstore.as_retriever(search_kwargs={"k": default_num_docs}),
            small=small
        )
        unique_docs = get_unique_docs(docs)
        context = "\n".join([doc.page_content for doc in unique_docs])

    return context.strip()