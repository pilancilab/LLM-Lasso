import argparse
import os

os.environ["OPENAI_API_KEY"] = "YOUR KEY HERE"
OPENROUTER_API_KEY = "YOUR KEY HERE"

openai_client = OpenAI()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-based feature selection for genes.")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the LLM model to use.")
    parser.add_argument("--genenames_path", type=str, required=True, help="Path to the gene names file (.pkl or .txt).")
    parser.add_argument("--k_min", type=int, default=0, help="Minimum number of top genes to select.")
    parser.add_argument("--k_max", type=int, default=50, help="Maximum number of top genes to select.")
    parser.add_argument("--step", type=int, default=5, help="Step size for the range of k values.")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for processing.")
    parser.add_argument("--category", type=str, required=True, help="Category for the query (e.g., cancer type).")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the results and scores.")
    parser.add_argument("--LLM_type", type=str, default="GPT", choices=["GPT", "General"],
                        help="Specify LLM type: 'GPT' for OpenAI or 'General' for OpenRouter.")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature for randomness in responses.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, default=0.9, help="Penalty for repeated tokens.")
    parser.add_argument("--wipe", action="store_true", help="If set, wipe the save directory before starting.")

    args = parser.parse_args()

    # Load gene names
    if args.genenames_path.endswith(".pkl"):
        with open(args.genenames_path, 'rb') as file:
            genenames = pkl.load(file)
    elif args.genenames_path.endswith(".txt"):
        with open(args.genenames_path, 'r') as file:
            genenames = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")

    # Call llm_score
    top_k_dict = llm_score(
        LLM_type=args.LLM_type,
        category=args.category,
        genenames=genenames,
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
        wipe=args.wipe
    )

    print("Top-k gene selection complete. Results saved.")