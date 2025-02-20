"""
Process PubMed RAG retrieval results through Langchain's PubMed tool focusing on retrieving:
(1). Information about each gene concerned;
(2). Information of the phenotype concerned in the domain of its known relation with certain genes;
(3). Interaction between each gene and each phenotype concerned.
"""
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from llm_lasso.utils.score_collection import *
from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory


pubmed_tool = PubmedQueryRun()

################################## Helper Functions ##################################
def parse_category_strings(input_string):
    """
    Parses a string containing two category names connected by 'and' into two separate strings.

    Args:
        input_string (str): Input string in the format "category1 and category2".

    Returns:
        tuple: A tuple containing the two parsed category strings.
    """
    # Split the string at 'and' and strip any extra whitespace
    parts = input_string.split(" and ")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    else:
        raise ValueError("Input string must contain exactly one 'and' separating two category names.")


def get_pubmed_prompts():
    ls = ["cat","gene","interact"]
    all_r = [] # retrieval
    all_s = [] # summary
    for item in ls:
        dir = f"prompts/pubmed_retrieval_prompt_{item}.txt"
        all_r.append(dir)
        dir = f"prompts/pubmed_summary_prompt_{item}.txt"
        all_s.append(dir)
    return all_r, all_s


def pubmed_retrieval_filter(text):
    """
    :param text: str
    :return: str or None
    """
    if text == "No good PubMed result was found":
        return ""
    else:
        return text


def fill_prompt(category, prompt_dir):
    with open(prompt_dir, "r", encoding="utf-8") as file:
        prompt_temp = file.read()
    temp = PromptTemplate(
        input_variables=["category"],  # Define variables to replace
        template=prompt_temp  # Pass the loaded template string
    )
    filled_prompt = temp.format(
        category=category,
    )
    return filled_prompt


def summarize_retrieval(
    gene: str,
    cat: str,
    ret: str,
    model: LLMQueryWrapperWithMemory,
    sum_prompt: str
):
    if ret == "":
        return ret
    
    if gene:
        full_prompt = f'{create_general_prompt(sum_prompt, cat, gene)} {ret}'
    else:
        full_prompt = f'{fill_prompt(cat, sum_prompt)} {ret}'
    
    return model.query(
        system_message="You are an expert in cancer genomics and bioinformatics.",
        full_prompt=full_prompt,
        sleep_time=1
    )

#################################### Main Function ##################################

# now assume binary classification - can extend to k class.
# collective retrieval function over three combinations: complexity = k+kg+g
def pubmed_retrieval(
    gene: str, category: str, model: LLMQueryWrapperWithMemory,
    retrieve_category = False, retrieve_genes = False,
    retrieve_interactions = True
):
    cat1, cat2 = parse_category_strings(category)
    s = []

    # We'll store retrieved documents here to detect duplicates
    seen_documents = set()
    seen_documents.add("")

    prompts_r, prompts_s = get_pubmed_prompts()
    for i, p in enumerate(prompts_r):
        if i == 0:
            if not retrieve_category:
                pass
            # Category prompts
            temp1 = fill_prompt(cat1, p)
            ret1 = pubmed_retrieval_filter(pubmed_tool.invoke(temp1))
            # Check if we have already seen this exact result
            if ret1 not in seen_documents:
                seen_documents.add(ret1)
                ret1_summary = summarize_retrieval(None, cat1, ret1, model, prompts_s[0])
                s.append(ret1_summary)

            temp2 = fill_prompt(cat2, p)
            ret2 = pubmed_retrieval_filter(pubmed_tool.invoke(temp2))
            if ret2 not in seen_documents:
                seen_documents.add(ret2)
                ret2_summary = summarize_retrieval(None, cat2, ret2, model, prompts_s[0])
                s.append(ret2_summary)

        elif i == 1:
            if not retrieve_genes:
                pass
            # Gene prompts
            for g_ in gene:
                temp1 = fill_prompt(g_, p)
                print("Pubmed: retrieving")
                ret1 = pubmed_retrieval_filter(pubmed_tool.invoke(temp1))
                if ret1 not in seen_documents:
                    seen_documents.add(ret1)
                    ret1_summary = summarize_retrieval(None, g_, ret1, model, prompts_s[1])
                    s.append(ret1_summary)

        else:
            if not retrieve_interactions:
                pass
            # Pair interaction (feature and class)
            for g_ in gene:
                temp1 = create_general_prompt(p, cat1, g_)
                ret1 = pubmed_retrieval_filter(pubmed_tool.invoke(temp1))
                if ret1 not in seen_documents:
                    seen_documents.add(ret1)
                    ret1_summary = summarize_retrieval(g_, cat1, ret1, model, prompts_s[2])
                    s.append(ret1_summary)

                temp2 = create_general_prompt(p, cat2, g_)
                ret2 = pubmed_retrieval_filter(pubmed_tool.invoke(temp2))
                if ret2 not in seen_documents:
                    seen_documents.add(ret2)
                    ret2_summary = summarize_retrieval(g_, cat2, ret2, model, prompts_s[2])
                    s.append(ret2_summary)

    # Return a joined string of all non-empty summaries
    return "\n\n".join(t for t in s if t)



# if __name__ == "__main__":
#     import constants
#     import os

#     os.environ["OPENAI_API_KEY"] = constants.OPENAI_API
#     input_str = "Acute myocardial infarction (AMI)  and diffuse large B-cell lymphoma (DLBCL)"
#     gene = ["AASS", "CLEC4D"]
#     print(pubmed_retrieval(gene, input_str, "gpt-4o"))