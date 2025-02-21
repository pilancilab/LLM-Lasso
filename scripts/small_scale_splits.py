import pandas as pd
from dataclasses import dataclass, field
from llm_lasso.data_splits import save_train_test_splits
from llm_lasso.utils.small_scale_data import process_spotify_csv, process_glioma, process_bank, load_uci_data, load_kaggle_data
from transformers.hf_argparser import HfArgumentParser

@dataclass
class Arguments:
    dataset: str = field(metadata={
        "help": "Which small-scale dataset to generate splits for.",
        "choices": ["Spotify", "Glioma", "Bank", "Wine", "Diabetes"]
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

    if args.dataset == "Spotify":
        X, y =  process_spotify_csv()
        save_train_test_splits(X, y, args.save_dir, n_splits=args.n_splits, seed=args.seed)
    elif args.dataset == "Glioma":
        X, y = process_glioma()
        save_train_test_splits(X, y, args.save_dir, n_splits=args.n_splits, seed=args.seed)
    elif args.dataset == "Bank":
        X, y = process_bank()
        save_train_test_splits(X, y, args.save_dir, n_splits=args.n_splits, seed=args.seed)
    elif args.dataset == "Wine":
        X, y, feat_names, _ = load_uci_data("Wine")
        X = pd.DataFrame(X, columns=feat_names)
        y = pd.Series(y)
        save_train_test_splits(X, y, args.save_dir, n_splits=args.n_splits, seed=args.seed)
    elif args.dataset == "Diabetes":
        X, y, feat_names, _ = load_kaggle_data("Diabetes")
        X = pd.DataFrame(X, columns=feat_names)
        y = pd.Series(y)
        save_train_test_splits(X, y, args.save_dir, n_splits=args.n_splits, seed=args.seed)
