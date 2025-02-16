# from langchain_community.chat_models import ChatOpenAI
from Expert_RAG.utils import *
from Expert_RAG.helper import *
import time
from Expert_RAG.omim_RAG_process import *
from Expert_RAG.pubMed_RAG_process import pubmed_retrieval

def get_rag_context(
    batch_genes, category, vectorstore, chat, rag_model_type="gpt-4o",
    pubmed_docs=False, filtered_cancer_docs = False, summarized_gene_docs = False,
    original_docs = True, orig_doc_k = 3, small = False
):
    context = ""
    skip_genes = set()
    if pubmed_docs:
        print("Retrieving pubmed")
        context += pubmed_retrieval(batch_genes, category, rag_model_type) + "\n"
        time.sleep(1)
    if filtered_cancer_docs:
        print("Retrieving cancer docs")
        (add_ctx, skip_genes) = get_filtered_cancer_docs_and_genes_found(
            batch_genes, vectorstore.as_retriever(search_kwargs={"k": 100}),
            chat, category, rag_model_type
        )
        print(add_ctx)
        context += add_ctx + "\n"
        time.sleep(1)
    if summarized_gene_docs:
        print("Retrieving gene docs")
        preamble = "\nAdditional gene information: \n" if context.strip() != "" else ""
        context += preamble + get_summarized_gene_docs(
            [gene for gene in batch_genes if gene not in skip_genes],
            chat
        ) + "\n"
        time.sleep(1)

    if original_docs:
        print("Retrieving original docs")
        assert (not pubmed_docs) and (not filtered_cancer_docs) and (not summarized_gene_docs)
        docs = retrieval_docs(
            batch_genes, category,
            vectorstore.as_retriever(search_kwargs={"k": orig_doc_k}),
            small=small
        )
        unique_docs = get_unique_docs(docs)
        context = "\n".join([doc.page_content for doc in unique_docs])

    return context.strip()