# Example usage
if __name__ == "__main__":
    input_str = "Acute myocardial infarction (AMI)  and diffuse large B-cell lymphoma (DLBCL)"
    gene = ["AASS", "CLEC4D"]
    # result1, result2 = parse_category_strings(input_str)
    # print(result1, result2)
    print(pubmed_retrieval(gene, input_str, "gpt-4o"))


if __name__ == "__main__":
    #main()

    # Create argument parser
    parser = argparse.ArgumentParser(description="Run feature selection baselines across a range of k values.")

    # Input arguments
    parser.add_argument("--split_dir", type=str, required=True, help="Path to the all of the train_test_splits")
    parser.add_argument("--n_splits", type=int, required=True, help="Number of splits in the dir")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results.")
    # Optional arguments
    parser.add_argument("--min", type=int, default=0, help="Minimum k value for feature selection.")
    parser.add_argument("--max", type=int, default=161, help="Maximum k value for feature selection.")
    parser.add_argument("--step", type=int, default=160, help="Step size for k values.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")

    # Parse the arguments
    args = parser.parse_args()

    (x_train, _, y_train, _) = read_train_test_splits(args.split_dir, args.n_splits)

    # Run the baseline function
    run_all_baselines_for_splits(
        x_train, y_train, args.save_dir, min=args.min, max=args.max, step=args.step, random_state=args.random_state)

    # Example Command
    """
    PYTHONPATH=$(pwd) python baselines/data-driven.py \
    --split_dir data/train_test_splits/FL \
    --n_splits 10 \
    --save_dir baselines/results/FL \
    --random_state 42
    """
