import sys
sys.path
sys.path.append('.')
import warnings
import constants
import os
from langchain_openai import OpenAIEmbeddings
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from langchain_community.vectorstores import Chroma
import pickle as pkl
from llm_lasso.adversarial.get_new_genenames import get_new_genenames
warnings.filterwarnings("ignore")  # Suppress warnings

@dataclass
class Arguments:
    feature_names_path: str = field(metadata={
        "help": "Path to the file containing the feature names (.pkl or .txt)"
    })
    fake_names_dir: str = field(metadata={
        "help": "Directory to store the adversarially-corrupted feature names"
    })
    category: str = field(default=None, metadata={
        "help": "Category for the query (e.g., cancer type)."
    })
    max_replace: int = field(default=500, metadata={
        "help": "Maximum number of features to adversarially corrupt."
    })
    replace_top: bool = field(default=False, metadata={
        "help": "Whether to replace the feature names based on prevalence in documents retrieved from the OMIM vectorstore (as opposed to randomly)."
    })

if __name__ == "__main__":
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]

    if args.feature_names_path.endswith(".pkl"):
        with open(args.feature_names_path, 'rb') as file:
            feature_names = pkl.load(file)
    elif args.feature_names_path.endswith(".txt"):
        with open(args.feature_names_path, 'r') as file:
            feature_names = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")
    
    if args.replace_top:
        embeddings = OpenAIEmbeddings()

        if os.path.exists(constants.OMIM_PERSIST_DIRECTORY):
            vectorstore = Chroma(persist_directory=constants.OMIM_PERSIST_DIRECTORY, embedding_function=embeddings)
        else:
            raise FileNotFoundError(f"Vector store not found at {constants.OMIM_PERSIST_DIRECTORY}. Ensure data is preprocessed and saved.")
    else:
        vectorstore = None

    get_new_genenames(
        genenames=feature_names,
        omim_api_key=constants.OMIM_KEYS[0],
        save_dir=args.fake_names_dir,
        max_replace=args.max_replace,
        replace_top_genes=args.replace_top,
        vectorstore=vectorstore,
        category=args.category,
    )