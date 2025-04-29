import sys
sys.path
sys.path.append('.')
from dataclasses import dataclass, field
from transformers.hf_argparser import HfArgumentParser
from llm_lasso.data_splits import read_train_test_splits
from llm_lasso.baselines.data_driven import feature_selector
import os
from tqdm import tqdm


@dataclass
class Arguments:
    split_dir: str = field(metadata={"help": "Path to all of the train_test_splits"})
    n_splits: int = field(metadata={"help": "Number of splits in the dir"})
    n: int = field(metadata={
        "help": "Number of features to preselect using a data-driven method"
    })
    save_dir: str = field(metadata={
        "help": "Directory to save the pre-selected genes"
    })
    method: str = field(default="mi", metadata={
        "help": "Data-driven method to use for feature pre-selection",
        "choices": ["mi", "mrmr", "xgboost", "rfe"]
    })
    random_state: int = field(default=42, metadata={"help": "Random seed for reproducibility."})


def main(args: Arguments):
    splits = read_train_test_splits(args.split_dir, args.n_splits)
    features = []

    for split in tqdm(splits):
        _, selected_features = feature_selector(
            split.x_train, split.y_train, method=args.method,
            k=args.n, random_state=args.random_state
        )
        features.append(sorted(selected_features))
    
    os.makedirs(args.save_dir, exist_ok=True)
    with open(f"{args.save_dir}/genes_per_split.txt", "w") as f:
            f.writelines([", ".join(line) + "\n" for line in features])


if __name__ == "__main__":
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)