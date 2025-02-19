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
    fake_names_path: str = field(metadata={
        "help": "Path to the adversarially-corrupted feature names (.pkl or .txt)"
    })
    category: str = field(metadata={
        "help": "Category for the query (e.g., cancer type)."
    })
    save_dir: str = field(metadata={
        "help": "Directory to save the results and scores."
    })
    experiment: str = field(default="llm-lasso", metadata={
        "help": "Whether to run llm-lasso or llm-score",
        "choices": ["llm-score", "llm-lasso"]
    })
    

    # LLM-score params
    k_min: int = field(default=0, metadata={
        "help": "Minimum number of top features that LLM-score selects."
    })
    k_max: int = field(default=50, metadata={
        "help": "Maximum number of top features that LLM-score selects."
    })
    step: int = field(default=5, metadata={
        "help": "Step size for the range of k values that LLM-score selects."
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
    
    # Load fake gene names
    if args.fake_names_path.endswith(".pkl"):
        with open(args.fake_names_path, 'rb') as file:
            fake_names = pkl.load(file)
    elif args.fake_names_path.endswith(".txt"):
        with open(args.fake_names_path, 'r') as file:
            fake_names = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")
    print(f'Total number of features in processing: {len(feature_names)}.')

    # Initialize LLM
    if llm_params.model_name == "gpt-4o":
        llm_type = LLMType.GPT4O
        api_key = constants.OPENAI_API
    elif llm_params.model_name == "o1":
        llm_type = LLMType.O1
        api_key = constants.OPENAI_API
    else:
        llm_type = LLMType.OPENROUTER
        api_key = constants.OPEN_ROUTER

    model = LLMQueryWrapperWithMemory(
        llm_type=llm_type,
        llm_name=llm_params.model_name,
        api_key=api_key,
        temperature=llm_params.temp,
        top_p=llm_params.top_p,
        repetition_penalty=llm_params.repetition_penalty
    )

    if args.experiment == "llm-lasso":
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

        if len(all_scores) != len(feature_names):
            raise ValueError(
                f"Mismatch between number of scores ({len(all_scores)}) and number of gene names ({len(feature_names)}).")

    else: # llm-select
        top_k_dict = llm_score(
            LLM_type=args.LLM_type,
            category=args.category,
            genenames=fake_names,
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
