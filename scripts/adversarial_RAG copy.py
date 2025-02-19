import os
import warnings
from tqdm import tqdm
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from Expert_RAG.utils import *
from Expert_RAG.helper import *
import Expert_RAG.data_processing as dp
import logging
import time
import json
from pydantic import BaseModel
from openai import OpenAI
from Expert_RAG.omim_RAG_process import *
from Expert_RAG.pubMed_RAG_process import pubmed_retrieval
from adversarial.utils import replace_top, replace_random, insert_fake_names_into_context
from llm_lasso.baselines.llm_score import llm_score
import argparse


warnings.filterwarnings("ignore")  # Suppress warnings
os.environ["OPENAI_API_KEY"] = "YOUR KEY HERE"

# Define paths and settings
persist_directory = "DATABASE DIRECTORY"  # Vector store path
data_file = "DATABASE DIRECTORY"  # Path to JSON data file
PERSIST = True  # Enable persistence

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize vector store and embeddings
if os.path.exists(persist_directory):
    print("Reusing existing combined database...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    raise FileNotFoundError(f"Vector store not found at {persist_directory}. Ensure data is preprocessed and saved.")

#retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # Retrieve top 8 most relevant chunks
llm_model = ChatOpenAI(model="gpt-4o", temperature=0.5)

openai_client = OpenAI()


class Score(BaseModel):
    gene: str
    penalty_factor: float
    reasoning: str


class GeneScores(BaseModel):
    scores: list[Score]


def wipe_RAG(save_dir):
    files_to_remove = [
        "results_RAG.txt",
        "gene_scores_RAG.pkl",
        "gene_scores_RAG.txt",
        "trial_scores_RAG.json",
    ]
    for file_name in files_to_remove:
        file_path = os.path.join(save_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG LLM-Lasso and LLM-Select with some fake gene names")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--experiment", type=str, choices=["llm-select", "llm-lasso"])
    parser.add_argument("--wipe", action="store_true", help="If set, wipe the save directory before starting.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the GPT model to use.")
    parser.add_argument("--genenames_path", type=str, required=True, help="Path to the gene names file (.pkl or .txt).")
    parser.add_argument("--fake_genenames_path", type=str, required=True, help="Path to the gene names file from get_new_genenames.py (.pkl or .txt).")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for processing.")
    parser.add_argument("--category", type=str, required=True, help="Category for the query (e.g., cancer type).")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the results and scores.")
    parser.add_argument("--LLM_type", type=str, default="GPT", choices=["GPT", "General"], help="Specify LLM type: 'GPT' for OpenAI or 'General' for OpenRouter.")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature for randomness in responses.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, default=0.9, help="Penalty for repeated tokens.")
    # LLM-SELECT arguments
    parser.add_argument("--k_min", type=int, default=0, help="Minimum number of top genes to select.")
    parser.add_argument("--k_max", type=int, default=50, help="Maximum number of top genes to select.")
    parser.add_argument("--step", type=int, default=5, help="Step size for the range of k values.")

    args = parser.parse_args()

    # Load gene names
    if args.genenames_path.endswith(".pkl"):
        with open(args.genenames_path, 'rb') as file:
            genenames = pkl.load(file)
    elif args.genenames_path.endswith(".txt"):
        with open(args.genenames_path, 'r') as file:
            genenames = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")
    
    if args.fake_genenames_path.endswith(".pkl"):
        with open(args.fake_genenames_path, 'rb') as file:
            new_genenames = pkl.load(file)
    elif args.fake_genenames_path.endswith(".txt"):
        with open(args.fake_genenames_path, 'r') as file:
            new_genenames = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")
    print(f'Total number of features in processing: {len(genenames)}.')

    if args.experiment == "llm-lasso":
        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()

        if os.path.exists(persist_directory):
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        else:
            raise FileNotFoundError(f"Vector store not found at {persist_directory}. Ensure data is preprocessed and saved.")

        if args.LLM_type == "GPT":
            if args.model_name == 'o1':
                raise NotImplementedError
            else:
                chat = ChatOpenAI(model=args.model_name, temperature=args.temp)
                results, all_scores = adversarial_hybrid_chain_GPT(
                    args.category, genenames, new_genenames, args.prompt_dir, args.save_dir, vectorstore,
                    chat, args.batch_size, args.n_trials, wipe = args.wipe,
                )
        elif args.LLM_type == "General":
            raise NotImplementedError
        print(f'Total number of scores collected: {len(all_scores)}.')
        # Save results
        os.makedirs(args.save_dir, exist_ok=True)
        results_file = os.path.join(args.save_dir, "results_RAG.txt")
        save_responses_to_file(results, results_file)

        scores_pkl_file = os.path.join(args.save_dir, "gene_scores_RAG.pkl")
        scores_txt_file = os.path.join(args.save_dir, "gene_scores_RAG.txt")
        save_scores_to_pkl(all_scores, scores_pkl_file)
        dp.convert_pkl_to_txt(scores_pkl_file, scores_txt_file)

        print(f"Results saved to {results_file}")
        print(f"Scores saved to {scores_pkl_file} and {scores_txt_file}")

        if len(all_scores) != len(genenames):
            raise ValueError(
                f"Mismatch between number of scores ({len(all_scores)}) and number of gene names ({len(genenames)}).")

    else: # llm-select
        top_k_dict = llm_score(
            LLM_type=args.LLM_type,
            category=args.category,
            genenames=new_genenames,
            prompt_dir=args.prompt_dir,
            save_dir=args.save_dir,
            model_name=args.model_name,
            k_min= int(args.k_min),
            k_max= int(args.k_max),
            step= int(args.step),
            temp=args.temp,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            batch_size=args.batch_size,
            n_trials=args.n_trials,
            wipe=args.wipe,
        )

        print("Top-k gene selection complete. Results saved.")