from typing import Any, Dict, List, Optional
import requests
import json
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from Expert_RAG import constants
import time

# Open Router codes
OPENROUTER_API_KEY = constants.OPEN_ROUTER

# Retry parameters
MAX_RETRIES = 3
RETRY_DELAY = 5  # in seconds

def query_openrouter(model, query, params=[1, 0.6, 1]):
    """
    Queries OpenRouter API with the given model and query, and returns the response.

    Parameters:
        model (str): The model to query (e.g., "meta-llama/llama-2-13b-chat").
        query (str): The question or query for the model.
        params (List[Int]): top_p, temperature, repetition_penalty

    Returns:
        str: The model's response content.
    """
    messages = [
        {"role": "system", "content": "You are an expert in cancer genomics and bioinformatics."},
        {"role": "user", "content": query}
    ]

    top_p, temp, rep = params

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": model,
                    "messages": messages,
                    "top_p": top_p,
                    "temperature": temp,
                    "repetition_penalty": rep,
                })
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "No content in response")
            else:
                raise ValueError(f"Error {response.status_code}: {response.text}")

        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("All retry attempts failed.")
                raise

# Creating the custom LLM class with Langchain
class OpenRouterLLM(LLM):
    """
    A custom LLM that interfaces with OpenRouter's Llama model.

    Example:

        .. code-block:: python

            model = OpenRouterLLM(api_key="your_api_key", model="meta-llama/llama-2-13b-chat")
            result = model.invoke("What are the key biomarkers for early-stage breast cancer?")
    """

    api_key: str
    """API key for OpenRouter access."""

    model: str
    """The model to query (e.g., "meta-llama/llama-2-13b-chat")."""

    top_p: float = 1.0
    """Top-p sampling parameter."""

    temperature: float = 0
    """Temperature for randomness in responses."""

    repetition_penalty: float = 1.0
    """Penalty for repeated tokens."""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Run the LLM on the given input using OpenRouter's API.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments.

        Returns:
            The model output as a string.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    data=json.dumps({
                        "model": self.model,
                        "messages": messages,
                        "top_p": self.top_p,
                        "temperature": self.temperature,
                        "repetition_penalty": self.repetition_penalty,
                        **kwargs
                    })
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                    # Handle stop tokens
                    if stop:
                        for token in stop:
                            content = content.split(token)[0]

                    return content
                else:
                    raise ValueError(f"Error {response.status_code}: {response.text}")

            except (requests.exceptions.RequestException, ValueError) as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("All retry attempts failed.")
                    raise

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty
        }

    @property
    def _llm_type(self) -> str:
        return "openrouter_llm"

# Example Usage
if __name__ == "__main__":
    model_name = "meta-llama/llama-3-8b-instruct"  # Specify the model
    query = "What are the key biomarkers for early-stage breast cancer?"
    params = [0.9, 0.9, 1]

    try:
        response_content = query_openrouter(model_name, query, params)
        print("Model Response:")
        print(response_content)
    except ValueError as e:
        print(e)

    # Test custom LLM
    OPENROUTER_API_KEY = constants.OPEN_ROUTER  # Replace with your actual API key
    model_name = "meta-llama/llama-3-8b-instruct"

    # Initialize the custom LLM
    llm = OpenRouterLLM(
        api_key=OPENROUTER_API_KEY,
        model=model_name,
        top_p=0.9,
        temperature=0,
        repetition_penalty=1.0
    )

    # # Test a single call
    # prompt = "What are the key biomarkers for early-stage breast cancer?"
    # try:
    #     response = llm.invoke(prompt)
    #     print(response)
    # except ValueError as e:
    #     print(e)
