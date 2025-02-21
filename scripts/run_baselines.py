
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass, field
from llm_lasso.data_splits import read_train_test_splits
from llm_lasso.baselines.data_driven import run_all_baselines_for_splits


@dataclass
class Arguments:
    split_dir: str = field(metadata={"help": "Path to all of the train_test_splits"})
    n_splits: int = field(metadata={"help": "Number of splits in the dir"})
    save_dir: str = field(metadata={"help": "Directory to save results."})
    min: int = field(default=0, metadata={"help": "Minimum k value for feature selection."})
    max: int = field(default=161, metadata={"help": "Maximum k value for feature selection."})
    step: int = field(default=160, metadata={"help": "Step size for k values."})
    random_state: int = field(default=42, metadata={"help": "Random seed for reproducibility."})

if __name__ == "__main__":
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]

    (x_train, _, y_train, _) = read_train_test_splits(args.split_dir, args.n_splits)

    args.max = min(args.max, len(x_train[0].columns)+1)
    args.step = min(args.step, args.max-args.min-1)

    # Run the baseline function
    run_all_baselines_for_splits(
        x_train, y_train, args.save_dir, min=args.min, max=args.max, step=args.step, random_state=args.random_state)
