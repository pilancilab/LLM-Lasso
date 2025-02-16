from omim_scrape.parse_omim import get_mim_number
import secrets
import numpy as np
import regex as re


def get_maybe_top_genes(genenames, category, retriever, min_doc_count=1):
    retrieval_prompt = f"""
    Retrieve information about {category}, especially in the context of genes' relevance to {category}.
    """
    docs = [str(doc).upper() for doc in retriever.get_relevant_documents(retrieval_prompt)]
    gene_to_count = {}
    for gene in genenames:
        gene_to_count[gene] = len([0 for doc in docs if gene.upper() in doc])
    return [x[0] for x in sorted(gene_to_count.items(), key=lambda x: x[1], reverse=True) if x[1] >= min_doc_count]


def get_fake_gene_names(n, min_len=4, max_len=6):
    genes = []
    while len(genes) < n:
        lengths = np.random.randint(min_len, max_len+1, size=n - len(genes))
        fake_data = secrets.token_urlsafe(np.sum(lengths)).upper()
        
        start = 0
        for ell in lengths:
            gene = fake_data[start:start+ell]
            if get_mim_number(gene, quiet=True) is None:
                genes.append(gene)
            start += ell
    return genes


def replace_top(genenames: list[str], category, retriever, max_replace=None,
                min_doc_count=1, min_gene_len=4, max_gene_len=6):
    if max_replace is None:
        max_replace = len(genenames)
    top_genes = set(get_maybe_top_genes(genenames, category, retriever, min_doc_count)[:max_replace])
    fake_names = get_fake_gene_names(len(top_genes), min_len=min_gene_len, max_len=max_gene_len)

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


def replace_random(genenames: list[str], n, min_gene_len=4, max_gene_len=6):
    idxs = np.random.choice(np.arange(len(genenames)), size=n, replace=False)
    fake_names = get_fake_gene_names(n, min_len=min_gene_len, max_len=max_gene_len)
    

    new_genenames = genenames[:]
    for (i, name) in zip(idxs, fake_names):
        if "|" in genenames[i]:
            continue
        new_genenames[i] = name
    return new_genenames


def insert_fake_names_into_context(true_genenames, new_genenames, context):
    for (old_name, new_name) in zip(true_genenames, new_genenames):
        if old_name == new_name:
            continue
        context = re.sub(old_name, new_name, context, flags=re.IGNORECASE)
    return context