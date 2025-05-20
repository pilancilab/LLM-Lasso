import sys
sys.path
sys.path.append('.')
import warnings
import constants
import os
from llm_lasso.baselines.lmpriors import query_lmpriors
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
import pickle as pkl
from argparse_helpers import LLMParams
warnings.filterwarnings("ignore")  # Suppress warnings

@dataclass
class Arguments:
    prompt_filename: str = field(metadata={
        "help": "Path to the prompt file."
    })
    feature_description_path: str = field(metadata={
        "help": "Path to the file containing the feature descriptions (.json)."
    })
    category: str = field(metadata={
        "help": "Category for the query (e.g., cancer type)."
    })
    data_name: str = field(metadata={
        "help": "Name of the dataset (e.g., 'Bank')."
    })
    save_dir: str = field(metadata={
        "help": "Directory to save the results and scores."
    })
    threshold: float = field(default=0.2, metadata={
        "help": "Threshold for the LMPriors to consider a feature as significant."
    })
    max_retries: int = field(default=5, metadata={
        "help": "Maximum number of retries for the LLM query."
    })
    display: bool = field(default=False, metadata={
        "help": "Display the results."
    })
    wipe: bool = field(default=False, metadata={
        "help": "Wipe the save directory before starting"})
    
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_API

    parser = HfArgumentParser([Arguments, LLMParams])
    (args, llm_params) = parser.parse_args_into_dataclasses()

    # Initialize LLM
    model = llm_params.get_model()

    # run lmpriors
    name = args.data_name
    save_dir = os.path.join(args.save_dir, name)

    query_lmpriors(args.prompt_filename,args.feature_description_path,save_dir, model, args.category, args.threshold, args.display, args.wipe, args.max_retries)

    print("LMPriors feature selection complete. Results saved.")