import sys
sys.path
sys.path.append('.')
import warnings
import constants
import os
from langchain_openai import OpenAIEmbeddings
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
    num_threads: int = field(default=1, metadata={
        "help": "number of threads to use for prompting the LLM for scores. Parallelization ocurrs across batches. Default is singlethreaded."
    })
    preselected_features_path: str = field(default=None, metadata={
        "help": "If applicable, .txt file output of preselect_genes_for_llm_lasso.py"
    }) 
    default_score:  float = field(default=1, metadata={
        "help": "Default score for, e.g., features not chosen in preselection."
    })


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_API

    parser = HfArgumentParser([PenaltyCollectionParams, Arguments, LLMParams])
    (penalty_params, args, llm_params) = parser.parse_args_into_dataclasses()

    # Load gene names
    preselection = False
    preselected_feature_names = None
    if args.preselected_features_path is not None:
        preselection = True
        preselected_feature_names = []
        with open(args.feature_names_path, "r") as file:
            for line in file.readlines():
                if "," not in line:
                    continue
                line = line.split(",")
                preselected_feature_names.append([x.strip() for x in line])
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
    model = llm_params.get_model()

    # Initialize embeddings and vector store
    if penalty_params.has_rag():
        embeddings = OpenAIEmbeddings()
        if os.path.exists(constants.OMIM_PERSIST_DIRECTORY):
            vectorstore = Chroma(persist_directory=constants.OMIM_PERSIST_DIRECTORY, embedding_function=embeddings)
        else:
            raise FileNotFoundError(f"Vector store not found at {constants.OMIM_PERSIST_DIRECTORY}. Ensure data is preprocessed and saved.")
    else:
        vectorstore = None

    results, all_scores = collect_penalties(
        category=args.category,
        feature_names=feature_names,
        prompt_file=args.prompt_filename,
        save_dir=args.save_dir,
        vectorstore=vectorstore,
        model=model,
        params=penalty_params,
        omim_api_key=constants.OMIM_KEYS[0],
        n_threads=args.num_threads,
        parallel=args.num_threads > 1,
        preselection_done=preselection,
        preselected_feature_names=preselected_feature_names,
        default_score=args.default_score
    )
    print(f'Total number of scores collected: {len(all_scores)}.')

    if len(all_scores) != len(feature_names):
        raise ValueError(
            f"Mismatch between number of scores ({len(all_scores)}) and number of gene names ({len(feature_names)}).")
