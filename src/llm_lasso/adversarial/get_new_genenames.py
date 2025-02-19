import os
import pickle as pkl
from llm_lasso.utils.omim import get_mim_number
import secrets
import numpy as np


def get_new_genenames(
    genenames, omim_api_key, save_dir, max_replace=200, replace_top_genes=True,
    vectorstore=None,category=None,  
):
    """
    Replaces genenames with random base64 strings, using OMIM to validate
    that the fake genenames are not actual genes, and saves the resulting
    list (with some real names and some adversarially-corrupted names) to
    a text and pickle file.

    Parameters:
    - `genenames`: list of original genenames.
    - `omim_api_key`: API key for checking OMIM to validate fake genenames.
    - `save_dir`: directory in which to save the new genenames.
    - `max_replace`: number of genenames to replace.
    - `replace_top_genes`: replace the genes that are most relevant to the
        category, as measured based on presence in documents retrieved by
        the OMIM vectorstore passed in. Otherwise, random genes are replaced.
    - `vectorstore`: OMIM vectorstore (for locating the top genes). Not
        required if `replace_top_genes` is `False`.
    - `category`: classification category (for locating the top genes with
        relevance to the category). Not required if `replace_top_genes` is
        `False`. 
    """
    if replace_top_genes:
        new_genenames = replace_top(
            genenames, omim_api_key, category,
            vectorstore.as_retriever(search_kwargs={"k": 2000}),
            max_replace, min_doc_count=0
        )
    else:
        new_genenames = replace_random(genenames, max_replace, omim_api_key)

    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + "/new_genenames.pkl", "wb") as f:
        pkl.dump(new_genenames, f)
    with open(save_dir + "/new_genenames.txt", "w") as f:
        f.write(str(new_genenames))

def get_maybe_top_genes(genenames, category, retriever, min_doc_count=1):
    """
    Retrieves the genes that appear most often in documents retrieved by the
    OMIM vectorstore, with respect to the category passed in.

    Returns a list of genenames that are present in at least `min_doc_count`
    documents, in reverse order of presence in documents.

    Parameters:
    - `genenames`: list of genenames to search for.
    - `category`: classification category (for locating the top genes with
        relevance to the category). Not required if `replace_top_genes` is
        `False`.
    - retriever: OMIM vectorstore, as a KNN retriever.
    - `min_doc_count`: only return genes that are present in this many documents.
    """
    retrieval_prompt = f"""
    Retrieve information about {category}, especially in the context of genes' relevance to {category}.
    """
    docs = [str(doc).upper() for doc in retriever.get_relevant_documents(retrieval_prompt)]
    gene_to_count = {}
    for gene in genenames:
        gene_to_count[gene] = len([0 for doc in docs if gene.upper() in doc])
    return [x[0] for x in sorted(gene_to_count.items(), key=lambda x: x[1], reverse=True) if x[1] >= min_doc_count]


def get_fake_gene_names(n, omim_api_key, min_len=4, max_len=6):
    """
    Produces `n` fake genenames (random base64 strings that are not valid
    OMIM gene names).

    Parameters:
    - `n`: number of fake genenames to generate.
    - `omim_api_key`: API key for checking OMIM to validate fake genenames.
    - `min_len`: minimum genename length.
    - `max_len`: maximum genename length.
    """
    genes = set()
    while len(genes) < n:
        lengths = np.random.randint(min_len, max_len+1, size=n - len(genes))
        fake_data = secrets.token_urlsafe(np.sum(lengths)).upper()
        
        start = 0
        for ell in lengths:
            gene = fake_data[start:start+ell]
            if get_mim_number(gene, omim_api_key, quiet=True) is None:
                genes.add(gene)
            start += ell
    return list(genes)


def replace_top(genenames: list[str], omim_api_key, category: str, retriever, max_replace=None,
                min_doc_count=1, min_gene_len=4, max_gene_len=6):
    """
    Replaces the top genes (based on presence in documents retreived by the
    `retriever` with respect to `category` genenames) with random base64
    strings, using OMIM to validate that the fake genenames are not actual genes.
    Returns the new list of genes.

    Parameters:
    - `genenames`: list of original genenames.
    - `omim_api_key`: API key for checking OMIM to validate fake genenames.
    - `category`: classification category (for locating the top genes with
        relevance to the category). Not required if `replace_top_genes` is
        `False`.
    - retriever: OMIM vectorstore, as a KNN retriever.
    - `max_replace`: number of genenames to replace.
    - `min_doc_count`: only return genes that are present in this many documents.
    - `min_gene_len`: minimum length of fake genenames.
    - `max_gene_len`: maximum length of fake genenames.
    """
    if max_replace is None:
        max_replace = len(genenames)
    top_genes = set(get_maybe_top_genes(genenames, category, retriever, min_doc_count)[:max_replace])
    fake_names = get_fake_gene_names(len(top_genes), omim_api_key, min_len=min_gene_len, max_len=max_gene_len)

    new_genenames = []
    i = 0
    for gene in genenames:
        if gene in top_genes and "|" not in gene:
            if fake_names[i][0] in ["_", "-"]:
                fake_names[i] = "0" + fake_names[i]
            new_genenames.append(fake_names[i])
            i += 1
        else:
            new_genenames.append(gene)
    return new_genenames


def replace_random(genenames: list[str], n, omim_api_key, min_gene_len=4, max_gene_len=6):
    """
    Replaces `n` random genes with random base64 strings, using OMIM to
    validate that the fake genenames are not actual genes.

    Returns the new list of genes.

    Parameters:
    - `genenames`: list of original genenames.
    - `n`: number of genes to replace.
    - `omim_api_key`: API key for checking OMIM to validate fake genenames.
    - `min_gene_len`: minimum length of fake genenames.
    - `max_gene_len`: maximum length of fake genenames..
    """
    idxs = np.random.choice(np.arange(len(genenames)), size=n, replace=False)
    fake_names = get_fake_gene_names(n, omim_api_key, min_len=min_gene_len, max_len=max_gene_len)
    

    new_genenames = genenames[:]
    for (i, name) in zip(idxs, fake_names):
        if "|" in genenames[i]:
            continue
        new_genenames[i] = name
    return new_genenames