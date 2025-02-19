"""
Stream Omim-based RAG method for LLM scores collection.
"""

import argparse
from Expert_RAG.hybrid_memory_GPT import *
import Expert_RAG.data_processing as dp
from LlaMa.lama import OpenRouterLLM
from LlaMa.RAG_lama import hybrid_chain_open
from langchain_core.rate_limiters import InMemoryRateLimiter

warnings.filterwarnings("ignore")  # Suppress warnings
os.environ["OPENAI_API_KEY"] = "YOUR KEY HERE" # Set OpenAI API key
OPENROUTER_API_KEY = "YOUR KEY HERE"  # Set OpenRouter API key

rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

def omim_RAG(
  prompt_dir, genenames_path, category, save_dir, LLM_type='GPT',
  model_name="gpt-4o", temp=0.5, top_p=0.9, repetition_penalty=0.9,
  batch_size=30, n_trials=1, wipe = False,
  summarized_gene_docs=False, filtered_cancer_docs=False, pubmed_docs=False, original_docs=True,
  original_rag_k=3, memory_size=500, skip_rag=False
):
    """
    Query genes in batches using memory-enhanced GPT or OpenRouter LLM responses.

    This function splits a list of gene names into batches, constructs prompts, and queries the selected LLM.
    It also maintains conversation memory and extracts the best gene-score pairs for each batch.

    Args:
        prompt_dir (str): Path to the prompt file used for constructing queries.
        genenames_path (str): Path to the file containing the list of gene names (.pkl or .txt format).
        category (str): The category or context for the query (e.g., "cancer type").
        save_dir (str): Directory where results and scores will be saved.
        LLM_type (str, optional): Choose between "GPT" (ChatOpenAI) or "General" (OpenRouter). Defaults to "GPT".
        model_name (str, optional): Name of the GPT model to use. Defaults to "gpt-4o".
        temp (float, optional): Temperature for randomness in responses. Defaults to 0.5.
        top_p (float, optional): Top-p sampling parameter. Defaults to 0.9.
        repetition_penalty (float, optional): Penalty for repeated tokens. Defaults to 0.9.
        batch_size (int, optional): Number of gene names to process per batch. Defaults to 30.
        n_trials (int, optional): Number of trials to run. Defaults to 1.

    Returns:
        tuple: Contains the following:
            - results (list[str]): Responses from the LLM for each batch.
            - best (list[tuple]): Best (gene, score) pairs for each batch.
            - batch_scores (list[list[float]]): Scores for all genes in each batch.
    """
    # Load gene names
    if genenames_path.endswith(".pkl"):
        with open(genenames_path, 'rb') as file:
            genenames = pkl.load(file)
    elif genenames_path.endswith(".txt"):
        with open(genenames_path, 'r') as file:
            genenames = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")

    print(f'Total number of features in processing: {len(genenames)}.')

    # Initialize embeddings and vector store
    persist_directory = "DATABASE DIR"
    embeddings = OpenAIEmbeddings()

    if os.path.exists(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        raise FileNotFoundError(f"Vector store not found at {persist_directory}. Ensure data is preprocessed and saved.")

    # Initialize the language model
    if LLM_type == "GPT":
        # if model_name == 'o1':
        #     chat = ChatOpenAI(model=model_name,  rate_limiter=rate_limiter)
        #     results, all_scores = hybrid_chain_o1(category, genenames, prompt_dir, save_dir, vectorstore, chat,
        #                                            batch_size, n_trials, wipe=wipe)
        # else:
        chat = ChatOpenAI(model=model_name, temperature=temp, rate_limiter=rate_limiter)
        results, all_scores = hybrid_chain_GPT(
            category, genenames, prompt_dir, save_dir, vectorstore, chat, batch_size, n_trials,
            wipe = wipe, summarized_gene_docs=summarized_gene_docs, filtered_cancer_docs=filtered_cancer_docs,
            pubmed_docs=pubmed_docs, original_docs=original_docs,
            original_rag_k=original_rag_k, memory_size=memory_size, model_type=model_name
        )
    elif LLM_type == "General":
        chat = OpenRouterLLM(
            api_key=OPENROUTER_API_KEY,
            model=model_name,
            top_p=top_p,
            temperature=temp,
            repetition_penalty=repetition_penalty,
            summarized_gene_docs=summarized_gene_docs, filtered_cancer_docs=filtered_cancer_docs,
            pubmed_docs=pubmed_docs, original_docs=original_docs,
            original_rag_k=original_rag_k, memory_size=memory_size
        )
        results, all_scores = hybrid_chain_open(category, genenames, prompt_dir, save_dir, vectorstore, chat, batch_size, n_trials, wipe = wipe)
    else:
        raise ValueError("LLM type should be either 'GPT', accessed through OpenAI, or 'General', accessed through OpenRouter.")

    print(f'Total number of scores collected: {len(all_scores)}.')
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    results_file = os.path.join(save_dir, "results_RAG.txt" if not skip_rag else "results_plain.txt")
    save_responses_to_file(results, results_file)

    scores_pkl_file = os.path.join(save_dir, "gene_scores_RAG.pkl" if not skip_rag else "gene_scores_plain.pkl")
    scores_txt_file = os.path.join(save_dir, "gene_scores_RAG.txt" if not skip_rag else "gene_scores_plain.txt")
    save_scores_to_pkl(all_scores, scores_pkl_file)
    dp.convert_pkl_to_txt(scores_pkl_file, scores_txt_file)

    print(f"Results saved to {results_file}")
    print(f"Scores saved to {scores_pkl_file} and {scores_txt_file}")

    if len(all_scores) != len(genenames):
        raise ValueError(
            f"Mismatch between number of scores ({len(all_scores)}) and number of gene names ({len(genenames)}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory-retained hybrid RAG GPT model.")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--wipe", action="store_true", help="If set, wipe the save directory before starting.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the GPT model to use.")
    parser.add_argument("--genenames_path", type=str, required=True, help="Path to the gene names file (.pkl or .txt).")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for processing.")
    parser.add_argument("--category", type=str, default="", help="Category for the query (e.g., cancer type).")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the results and scores.")
    parser.add_argument("--LLM_type", type=str, default="GPT", choices=["GPT", "General"], help="Specify LLM type: 'GPT' for OpenAI or 'General' for OpenRouter.")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature for randomness in responses.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, default=0.9, help="Penalty for repeated tokens.")
    parser.add_argument("--get_pubmed_docs", action="store_true")
    parser.add_argument("--get_cancer_docs", action="store_true")
    parser.add_argument("--get_gene_docs", action="store_true")
    parser.add_argument("--skip_rag", action="store_true")
    parser.add_argument("--original_rag_k", type=int, default=3)
    parser.add_argument("--memory_size", type=int, default=500)

    args = parser.parse_args()

    original_rag = not (args.skip_rag or args.get_pubmed_docs or args.get_cancer_docs or args.get_gene_docs)
    if args.skip_rag:
        assert not args.get_pubmed_docs and not args.get_cancer_docs and not args.get_gene_docs

    omim_RAG(
        prompt_dir=args.prompt_dir,
        genenames_path=args.genenames_path,
        category=args.category,
        save_dir=args.save_dir,
        LLM_type=args.LLM_type,
        model_name=args.model_name,
        temp=args.temp,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        n_trials=args.n_trials,
        wipe=args.wipe,
        pubmed_docs=args.get_pubmed_docs,
        filtered_cancer_docs=args.get_cancer_docs,
        summarized_gene_docs=args.get_gene_docs,
        original_docs=original_rag,
        original_rag_k=args.original_rag_k,
        memory_size=args.memory_size,
        skip_rag=args.skip_rag
    )