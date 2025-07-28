import os
from pydantic import BaseModel
import requests
import tiktoken
from llm_lasso.llm_penalty.llm import LLMType, OPENAI_TYPES, LLMQueryWrapperWithMemory


OPENAI_PRICES = {
    "gpt-4.1": {
        "Input": 2.0,
        "Cached input": 0.5,
        "Output": 8.0
    },
    "gpt-4o": {
        "Input": 2.5,
        "Cached input": 1.25,
        "Output": 10.0
    },
    "gpt-4o-mini": {
        "Input": 0.15,
        "Cached input": 0.075,
        "Output": 0.6
    },
    "o1": {
        "Input": 15.0,
        "Cached input": 7.5,
        "Output": 60.0
    },
    "o3": {
        "Input": 2.0,
        "Cached input": 0.5,
        "Output": 8.0
    },
    "o4-mini": {
        "Input": 1.1,
        "Cached input": 0.275,
        "Output": 4.4
    },
    "o3-mini": {
        "Input": 1.1,
        "Cached input": 0.55,
        "Output": 4.4
    },
    "o1-mini": {
        "Input": 1.1,
        "Cached input": 0.55,
        "Output": 4.4
    },
}


def get_pricing(model_provider):
    assert model_provider in ["openai", "openrouter"]
    if model_provider == "openai":
        return OPENAI_PRICES
    
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"
    }

    response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)

    data = response.json()
    pricing = {}
    for model in data["data"]:
        pricing = model.get("pricing")
        pricing["prompt"] = float(pricing["prompt"])
        pricing["completion"] = float(pricing["completion"])
        pricing = {
            "Input":  round(pricing["prompt"] if pricing["prompt"] > 0.001 else pricing["prompt"] * 1e6, 10),
            "Output":  round(pricing["completion"] if pricing["completion"] > 0.001 else pricing["completion"] * 1e6, 10)
        }
        pricing[model["id"]] = pricing
    return pricing


def count_tokens(content, model: str):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    if type(content) is str:
        return len(encoding.encode(content))
    n_tokens = 0
    for message in content:
        if message["type"] == "text":
            n_tokens += len(encoding.encode(message["text"]))
        else:
            raise NotImplementedError(f"Unsupported message type: {message['type']}")
    return n_tokens


class LLMQueryWithPricing(LLMQueryWrapperWithMemory):
    def __init__(
        self,
        llm_type: int,
        llm_name: str,
        api_key: str,
        temperature: float = 0.5,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
        super().__init__(
            llm_type, llm_name, api_key,
            temperature, top_p, repetition_penalty
        )
        self.llm_name = llm_name
        self.pricing_dict = get_pricing(
            "openai" if llm_type in OPENAI_TYPES else "openrouter"
        )
    
    def structured_query(
        self, system_message, full_prompt, response_format_class,
        sleep_time: 0.1
    ):
        """
        Perform a query with a structured (i.e., python class) output
        """
        result: BaseModel = super().structured_query(
            system_message, full_prompt, response_format_class, sleep_time
        )

        input_tokens = count_tokens(
            f"{system_message}\n{self._maybe_get_memory()}\n\n{full_prompt}",
            self.llm_name
        )
        output_tokens = count_tokens(result.model_dump_json(), self.llm_name)
        price = self.pricing_dict[self.llm_name]["Input"] * input_tokens / 1e6 + \
            self.pricing_dict[self.llm_name]["Output"] * output_tokens / 1e6
       
        return result, price
    
    def query(self, system_message, full_prompt, sleep_time: 0.1):
        """
        Query the LLM with a system message and user prompt
        """

        result = super().query(
            system_message, full_prompt, sleep_time
        )

        full_prompt = f"{self._maybe_get_memory()}\n\n{full_prompt}"
        input_tokens = count_tokens(full_prompt, self.llm_name)
        output_tokens = count_tokens(result.model_dump_json(), self.llm_name)
        price = self.pricing_dict[self.llm_name]["Input"] * input_tokens / 1e6 + \
            self.pricing_dict[self.llm_name]["Output"] * output_tokens / 1e6

        return result, price

    def retry_last(self, sleep_time=0.1):
        """
        Retry the latest query
        """
        if self.last is None:
            return ""

        print("Retrying latest query...")
        if self.last.sructured:
            return self.structured_query(
                self.last.system_message,
                self.last.prompt,
                self.last.format_class,
                sleep_time
            )

        return self.query(
            self.last.system_message,
            self.last.prompt,
            sleep_time
        )