"""
Process PubMed RAG retrieval results through Langchain's PubMed tool.
"""
import os
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.schema import SystemMessage, HumanMessage
from Expert_RAG.utils import *
from openai import OpenAI
import time
from langchain_core.rate_limiters import InMemoryRateLimiter


rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

os.environ["OPENAI_API_KEY"] = "YOUR API KEY HERE"  # Set OpenAI API key
client = OpenAI()
llm = ChatOpenAI(model="gpt-4o", temperature=0.5, rate_limiter=rate_limiter)
tool = PubmedQueryRun()
#ret_prompt = "prompts/pubmed_retrieval_prompt_cat.txt"
#sum_prompt = "prompts/pubmed_summary_prompt_cat.txt"

# Helpers
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
    if text == "No good PubMed Result was found":
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

def summarize_retrieval(gene, cat, ret, model, sum_prompt):
    time.sleep(1)
    print("Pubmed: Summarizing")
    if ret == "":
        return ret
    if gene:
        full_prompt = f'{create_general_prompt(sum_prompt, cat, gene)} {ret}'
    else:
        full_prompt = f'{fill_prompt(cat, sum_prompt)} {ret}'
    if ret != "":
        doc1 = Document(page_content=ret)
        if model == "o1":
            msg1 = [
                {
                    "role": "assistant",
                    "content": full_prompt
                }
            ]
            completion = client.chat.completions.create(
                model="o1",
                messages=msg1
            )
            ret = completion.choices[0].message.content # update ret1
        elif model == "gpt-4o":
            msg1 = [
                SystemMessage(content="You are an expert in cancer genomics and bioinformatics."),
                HumanMessage(content=full_prompt)
            ]
            ret = llm.invoke(msg1).content
        else:
            print("Model must either be 'gpt-4o' or 'o1'.")
            sys.exit(-1)
    return ret

# now assume binary classification - can extend to k class.
# collective retrieval function over three combinations: complexity = k+kg+g
def pubmed_retrieval(
    gene, category, model, retrieve_category = False,
    retrieve_genes = False, retrieve_interactions = True
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
            ret1 = pubmed_retrieval_filter(tool.invoke(temp1))
            # Check if we have already seen this exact result
            if ret1 not in seen_documents:
                seen_documents.add(ret1)
                ret1_summary = summarize_retrieval(None, cat1, ret1, model, prompts_s[0])
                s.append(ret1_summary)

            temp2 = fill_prompt(cat2, p)
            ret2 = pubmed_retrieval_filter(tool.invoke(temp2))
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
                ret1 = pubmed_retrieval_filter(tool.invoke(temp1))
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
                ret1 = pubmed_retrieval_filter(tool.invoke(temp1))
                if ret1 not in seen_documents:
                    seen_documents.add(ret1)
                    ret1_summary = summarize_retrieval(g_, cat1, ret1, model, prompts_s[2])
                    s.append(ret1_summary)

                temp2 = create_general_prompt(p, cat2, g_)
                ret2 = pubmed_retrieval_filter(tool.invoke(temp2))
                if ret2 not in seen_documents:
                    seen_documents.add(ret2)
                    ret2_summary = summarize_retrieval(g_, cat2, ret2, model, prompts_s[2])
                    s.append(ret2_summary)

    # Return a joined string of all non-empty summaries
    return "\n\n".join(t for t in s if t)

# store pubmed retrieval in a hashmap to avoid duplication.



# Example usage
if __name__ == "__main__":
    input_str = "Acute myocardial infarction (AMI)  and diffuse large B-cell lymphoma (DLBCL)"
    gene = ["AASS", "CLEC4D"]
    # result1, result2 = parse_category_strings(input_str)
    # print(result1, result2)
    print(pubmed_retrieval(gene, input_str, "gpt-4o"))