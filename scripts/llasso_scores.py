import sys
sys.path
sys.path.append('.')
import warnings
import constants
import os
from langchain_openai import OpenAIEmbeddings
from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory, LLMType
from llm_lasso.llm_penalty.penalty_collection import collect_penalties, PenaltyCollectionParams
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from langchain_community.vectorstores import Chroma
import pickle as pkl
from argparse_helpers import LLMParams
warnings.filterwarnings("ignore")  # Suppress warnings

@dataclass
class Arguments:
    prompt_filename: str = field(metadata={
        "help": "Path to the prompt file."
    })
    feature_names_path: str = field(metadata={
        "help": "Path to the file containing the feature names (.pkl or .txt)"
    })
    category: str = field(metadata={
        "help": "Category for the query (e.g., cancer type)."
    })
    save_dir: str = field(metadata={
        "help": "Directory to save the results and scores."
    })


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_API

    parser = HfArgumentParser([PenaltyCollectionParams, Arguments, LLMParams])
    (penalty_params, args, llm_params) = parser.parse_args_into_dataclasses()

    # Load gene names
    if args.feature_names_path.endswith(".pkl"):
        with open(args.feature_names_path, 'rb') as file:
            feature_names = pkl.load(file)
    elif args.feature_names_path.endswith(".txt"):
        with open(args.feature_names_path, 'r') as file:
            feature_names = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")
    
    print(f'Total number of features in processing: {len(feature_names)}.')

    # Initialize LLM
    if llm_params.model_type == "gpt-4o":
        llm_type = LLMType.GPT4O
        api_key = constants.OPENAI_API
    elif llm_params.model_type == "o1":
        llm_type = LLMType.O1
        api_key = constants.OPENAI_API
    else:
        llm_type = LLMType.OPENROUTER
        api_key = constants.OPEN_ROUTER

    model = LLMQueryWrapperWithMemory(
        llm_type=llm_type,
        llm_name=llm_params.get_model_name(),
        api_key=api_key,
        temperature=llm_params.temp,
        top_p=llm_params.top_p,
        repetition_penalty=llm_params.repetition_penalty
    )

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    if os.path.exists(constants.OMIM_PERSIST_DIRECTORY):
        vectorstore = Chroma(persist_directory=constants.OMIM_PERSIST_DIRECTORY, embedding_function=embeddings)
    else:
        raise FileNotFoundError(f"Vector store not found at {constants.OMIM_PERSIST_DIRECTORY}. Ensure data is preprocessed and saved.")

    results, all_scores = collect_penalties(
        args.category, feature_names,
        args.prompt_filename,
    )
    print(f'Total number of scores collected: {len(all_scores)}.')

    if len(all_scores) != len(feature_names):
        raise ValueError(
            f"Mismatch between number of scores ({len(all_scores)}) and number of gene names ({len(feature_names)}).")
