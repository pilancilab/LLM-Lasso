import sys
sys.path
sys.path.append('.')
import warnings
import constants
import os
from llm_lasso.baselines.llm_score import llm_score
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
    feature_names_path: str = field(metadata={
        "help": "Path to the file containing the feature names (.pkl or .txt)"
    })
    category: str = field(metadata={
        "help": "Category for the query (e.g., cancer type)."
    })
    save_dir: str = field(metadata={
        "help": "Directory to save the results and scores."
    })

    batch_size: int = field(default=30, metadata={
        "help": "Number of genes to pass into the LLM at once"})
    n_trials: int = field(default=1, metadata={
        "help": "Number of trials to average over"})
    wipe: bool = field(default=False, metadata={
        "help": "Wipe the save directory before starting"})
    
    k_min: int = field(default=0, metadata={
        "help": "Minimum number of top genes to select."
    })
    k_max: int = field(default=50, metadata={
        "help": "Maximum number of top genes to select."
    })
    step: int = field(default=5, metadata={
        "help": "Step size for the range of k values."
    })

    def clamp_kmax_and_step(self, n_features):
        self.k_max = min(self.k_max, n_features)
        self.step = min(self.step, self.k_max - self.k_min - 1)


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_API

    parser = HfArgumentParser([Arguments, LLMParams])
    (args, llm_params) = parser.parse_args_into_dataclasses()

    # Load gene names
    if args.feature_names_path.endswith(".pkl"):
        with open(args.feature_names_path, 'rb') as file:
            feature_names = pkl.load(file)
    elif args.feature_names_path.endswith(".txt"):
        with open(args.feature_names_path, 'r') as file:
            feature_names = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")
    args.clamp_kmax_and_step(len(feature_names))
    
    print(f'Total number of features in processing: {len(feature_names)}.')

    # Initialize LLM
    model = llm_params.get_model()

    top_k_dict = llm_score(
        category=args.category,
        feature_names=feature_names,
        prompt_filename=args.prompt_filename,
        save_dir=args.save_dir,
        model=model,
        batch_size=args.batch_size,
        n_trials=args.n_trials,
        wipe=args.wipe,
        k_min=args.k_min,
        k_max=args.k_max,
        step=args.step
    )

    print("Top-k gene selection complete. Results saved.")
