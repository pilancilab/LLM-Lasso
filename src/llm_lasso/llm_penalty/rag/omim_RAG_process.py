"""
This script offers two additional features to process the retrieval information from OMIM in the RAG pipeline:
(1). Optional summarization of the retrieved documents.
(2). Filtering of the retrieved documents based on the presence of gene names.
"""
from llm_lasso.utils.omim import get_mim_number, fetch_omim_data, parse_omim_response
from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory


def get_summarized_gene_docs(
    batch_genes: list[str], model: LLMQueryWrapperWithMemory, omim_api_key: str,
) -> str:
    docs = [
        fetch_omim_data(get_mim_number(gene, omim_api_key), omim_api_key)
        for gene in batch_genes
    ]
    concat_docs = "\n".join([str(parse_omim_response(doc)) for doc in docs if doc is not None])

    prompt = """
    Faithfully and briefly summarize each of following documents one by one, in the order presented.
    Do not skip documents or add any of your own commentary.
    """

    return model.query(
        system_message="You are an expert assistant with access to gene and cancer knowledge.",
        full_prompt=f"{prompt}\n{concat_docs}",
        sleep_time=1
    )


def get_filtered_cancer_docs_and_genes_found(
    batch_genes: list[str],
    retriever, # VectorStoreRetriver
    model: LLMQueryWrapperWithMemory,
    category: str,
) -> tuple[str, set[str]]:

    retrieval_prompt = f"""
    Retrieve information about {category}, especially in the context of genes' relevance to {category}.
    """

    summarization_prompt = f"""
    Faithfully and briefly summarize the following text with a focus on important details on genes' relation to {category}.
    Do not skip documents or add any of your own commentary.
    """

    docs = retriever.get_relevant_documents(retrieval_prompt)
    docs_w_genes = [
        doc for doc in docs 
            if any([gene.upper() in str(doc).upper() for gene in batch_genes])
    ]

    genes = set([doc.metadata['gene_name'] for doc in docs_w_genes])
    docs = "\n".join([f"ARTICLE {i}: " + doc.page_content for (i, doc) in enumerate(docs)])

    ret = model.query(
        system_message="You are an expert in cancer genomics and bioinformatics.",
        full_prompt=f"{summarization_prompt}\n{docs}",
        sleep_time=1
    )

    return (ret, genes)