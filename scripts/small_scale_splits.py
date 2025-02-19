from dataclasses import dataclass, field
from llm_lasso.data_splits import save_train_test_splits
from llm_lasso.utils.small_scale_data import process_spotify_csv
from transformers.hf_argparser import HfArgumentParser

@dataclass
class Arguments:
    dataset: str = field(metadata={
        "help": "Which small-scale dataset to generate splits for.",
        "choices": ["spotify"]
    })
    save_dir: str = field(metadata={
        "help": "Directory in which to save the splits."
    })
    n_splits: int = field(default=10, metadata={
        "help": "Number of different train/test splits to generate."
    })
    seed: int = field(default=42, metadata={
        "help": "Random seed"
    })

if __name__ == "__main__":
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]

    if args.dataset == "spotify":
        X, y =  process_spotify_csv()
        save_train_test_splits(X, y, args.save_dir, n_splits=args.n_splits, seed=args.seed)
