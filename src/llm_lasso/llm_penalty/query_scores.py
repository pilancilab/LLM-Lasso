from pydantic import BaseModel
from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory
from llm_lasso.llm_penalty.llm_with_pricing import LLMQueryWithPricing
from llm_lasso.utils.score_collection import extract_scores_from_responses
import logging

class Score(BaseModel):
    gene: str
    penalty_factor: float
    reasoning: str


class GeneScores(BaseModel):
    scores: list[Score]


def query_scores_with_retries(
    model: LLMQueryWithPricing,
    system_message: str,
    full_prompt: str,
    batch_features: list[str],
    retry_limit=50
) -> tuple[list[int], str]:
    """
    Query an LLM for feature scores, with automatic retries.
    """
    upper_batch_names = [n.upper() for n in batch_features]

    total_price = 0
    if model.has_structured_output():
        gene_scores, price = model.structured_query(
            system_message=system_message,
            full_prompt=full_prompt,
            response_format_class=GeneScores,
            sleep_time=1,
        )
        print(gene_scores)
        total_price += price

        scores_list = [score for score in gene_scores.scores if score.gene.upper() in upper_batch_names]
        features_retrieved = set([score.gene.upper() for score in scores_list])
        missing = set(upper_batch_names).difference(features_retrieved)

        # Retry logic for score validation
        n_retries = 0
        while len(missing) > 0:
            logging.warning(f"We are missing genes {missing}")
            assert n_retries < retry_limit
            n_retries += 1

            gene_scores, price = model.retry_last(sleep_time=1)
            total_price += price
            scores_list = [score for score in gene_scores.scores if score.gene.upper() in upper_batch_names]
            features_retrieved = set([score.gene.upper() for score in scores_list])
            missing = set(upper_batch_names).difference(features_retrieved)
        
        genes_to_scores = {
            score.gene: score.penalty_factor for score in gene_scores.scores
        }
        batch_scores_partial = [genes_to_scores[feature] for feature in batch_features]
        output = gene_scores.model_dump_json()
    else:
        output, price = model.query(
            system_message=system_message,
            full_prompt=full_prompt,
            sleep_time=1,
        )
        total_price += price

        batch_scores_partial = extract_scores_from_responses(
            output if isinstance(output, list) else [output],
            batch_features
        )

        # Retry logic for score validation
        n_retries = 0
        while len([score for score in batch_scores_partial if score is not None]) != len(batch_features):
            logging.info(output)
            assert n_retries < retry_limit
            n_retries += 1
            try:
                logging.warning(f"Batch scores count mismatch for genes {batch_features}. Retrying...")
                output, price = model.retry_last(sleep_time=1)
                total_price += price
                batch_scores_partial = extract_scores_from_responses(
                    output if isinstance(output, list) else [output],
                    batch_features
                )
            except Exception as e:
                logging.error(f"Error during retry: {str(e)}. Continuing retry...")
        # end retry while loop
    # end structured output if/else
    return batch_scores_partial, output, total_price