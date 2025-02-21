import os
import warnings
from llm_lasso.utils.score_collection import *
from llm_lasso.llm_penalty.rag.pubMed_RAG_process import pubmed_retrieval
from transformers.hf_argparser import HfArgumentParser
from argparse_helpers import LLMParams
from dataclasses import dataclass, field
import constants

warnings.filterwarnings("ignore")  # Suppress warnings


@dataclass
class Arguments:
    gene: str = field(metadata={
        "help": "Which list of genes to query Pub-Med for.",
        "example": ["AASS", "CLEC4D"]
    })
    category: str = field(metadata={
        "help": "Which categories to query Pub-Med for.",
        "example": "Acute myocardial infarction (AMI)  and diffuse large B-cell lymphoma (DLBCL)"
    })
    retrieve_category: bool = field(default=10, metadata={
        "help": "Whether to retrieve Pub-Med information by category queried."
    })
    retrieve_genes: bool = field(default=42, metadata={
        "help": "Whether to retrieve Pub-Med information by genes queried."
    })
    retrieve_interactions: bool = field(default=42, metadata={
        "help": "Whether to retrieve Pub-Med information by interaction/relation of the queried gene and category pair."
    })

# Example usage
if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_API
    parser = HfArgumentParser([Arguments, LLMParams])
    (args, llm_params) = parser.parse_args_into_dataclasses()
    # Initialize LLM
    model = llm_params.get_model()
    result = pubmed_retrieval(
        gene=args.gene,
        category=args.category,
        model=model,
        retrieve_category=args.retrieve_category,
        retrieve_genes=args.retrieve_genes,
        retrieve_interactions=args.retrieve_interactions
    )
    print(result)

