"""
This script offers two additional features to process the retrieval information from OMIM in the RAG pipeline:
(1). Optional summarization of the retrieved documents.
(2). Filtering of the retrieved documents based on the presence of gene names.
"""

import pickle as pkl
from omim_scrape.process_mimNumber import get_mim_number
from omim_scrape.parse_omim import fetch_omim_data_with_key, parse_omim_response
from langchain.schema import SystemMessage, HumanMessage
import time
import sys


def get_summarized_gene_docs(batch_genes, chat) -> str:
    docs = [fetch_omim_data(get_mim_number(gene)) for gene in batch_genes]
    concat_docs = "\n".join([str(parse_omim_response(doc)) for doc in docs if doc is not None])

    prompt = """
    Faithfully and briefly summarize each of following documents one by one, in the order presented.
    Do not skip documents or add any of your own commentary.
    """

    messages = [
        SystemMessage(content="You are an expert assistant with access to gene and cancer knowledge."),
        HumanMessage(content=prompt + "\n" + concat_docs)
    ]
    response = chat(messages)
    return response.content


def get_filtered_cancer_docs_and_genes_found(
    batch_genes, retriever, chat, category,
    rag_model_type="gpt-4o"
) -> tuple[str, set[str]]:
    time.sleep(1)
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

    if rag_model_type == "o1":
        msg1 = [
            {
                "role": "assistant",
                "content": summarization_prompt + "\n" + docs
            }
        ]
        completion = chat.completions.create(
            model="o1",
            messages=msg1
        )
        ret = completion.choices[0].message.content # update ret1
    elif rag_model_type == "gpt-4o":
        msg1 = [
            SystemMessage(content="You are an expert in cancer genomics and bioinformatics."),
            HumanMessage(content=summarization_prompt + "\n" + docs)
        ]
        ret = chat(msg1).content
    else:
        print("Model must either be 'gpt-4o' or 'o1'.")
        sys.exit(-1)
    return (ret, genes)