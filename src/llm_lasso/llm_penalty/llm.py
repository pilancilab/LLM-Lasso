from enum import IntEnum
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLM
from openai import OpenAI
import requests
import json
import time
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationSummaryBufferMemory
from dataclasses import dataclass, field


# Retry parameters
MAX_RETRIES = 3
RETRY_DELAY = 5  # in seconds


class LLMType(IntEnum):
    GPT4O = 0
    O1 = 1
    OPENROUTER = 2


@dataclass
class CachedQuery:
    system_message: str
    prompt: str
    sructured: bool = field(default=False)
    format_class: any = field(default=None)


class LLMQueryWrapperWithMemory:
    def __init__(
        self,
        llm_type: int,
        llm_name: str,
        api_key: str,
        temperature: float = 0.5,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ):
        self.llm_type = llm_type
        self.llm_name = llm_name
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        if llm_type == LLMType.GPT4O or llm_type == LLMType.O1:
            self.chat = ChatOpenAI(model=llm_name, temperature=temperature)
            self.openai_client = OpenAI()
        else:
            self.llm_type = LLMType.OPENROUTER
            self.chat = OpenRouterLLM(
                api_key=api_key,
                model=llm_name,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

        self.memory = None
        self.last_query = None

    def start_memory(
        self, memory_size, remember_outputs=True,
        default_output="Processing prompt, continuing from previous prompts"
    ):
        self.remember_outputs = remember_outputs
        self.default_output = default_output
        if self.llm_type != LLMType.O1:
            self.memory = ConversationSummaryBufferMemory(llm=self.chat, max_token_limit=memory_size)
            self.memory.clear()

    def disable_memory(self):
        self.memory = None

    def has_structured_output(self):
        return self.llm_type == LLMType.GPT4O or self.llm_type == LLMType.O1
    
    def _maybe_get_memory(self):
        if self.memory == None:
            return ""
        memory_context = self.memory.load_memory_variables({})
        return memory_context.get("history", "")
    
    def maybe_add_to_memory(self, query, output):
        if self.memory == None:
            return
        if not self.remember_outputs:
            output = self.default_output
        self.memory.save_context({"input": query}, {"output": output})
    
    def structured_query(
        self, system_message, full_prompt, response_format_class,
        sleep_time: 0.1
    ):
        assert self.has_structured_output()
        time.sleep(sleep_time)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{self._maybe_get_memory()}\n\n{full_prompt}"}
        ]

        self.last = CachedQuery(
            system_message=system_message,
            prompt=full_prompt,
            sructured=True,
            response_format_class=response_format_class
        )

        if self.llm_type != LLMType.O1:
            completion = self.openai_client.beta.chat.completions.parse(
                model=self.llm_name,
                messages=messages,
                response_format=response_format_class,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else: # o1 does not support temperature and top-p
            completion = self.openai_client.beta.chat.completions.parse(
                model=self.llm_name,
                messages=messages,
                response_format=response_format_class
            )
        return completion.choices[0].message.parsed
    
    def query(self, system_message, full_prompt, sleep_time: 0.1):
        time.sleep(sleep_time)

        self.last = CachedQuery(
            system_message=system_message,
            prompt=full_prompt,
            sructured=False,
        )

        full_prompt = f"{self._maybe_get_memory()}\n\n{full_prompt}"
        if self.llm_type == LLMType.O1:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": full_prompt}
            ]
            completion = self.chat.completions.create(
                model="o1",
                messages=messages
            )
            output = completion.choices[0].message.content
        elif self.llm_type == LLMType.GPT4O:
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=full_prompt)
            ]
            response = self.chat(messages)
            output = response.content
        else:
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=full_prompt)
            ]

            # Serialize messages into a single string
            serialized_prompt = "\n".join([f"{msg.content}" for msg in messages])

            # Query OpenRouter
            output = self.chat(serialized_prompt)

        return output

    def maybe_retry_last(self, sleep_time=0.1):
        if self.last is None:
            return ""

        print("Retrying...")
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

    temperature: float = 0.0
    """Temperature for randomness in responses."""

    repetition_penalty: float = 1.0
    """Penalty for repeated tokens."""

    def _call(
        self,
        prompt: str,
        stop: list[str] = None,
        **kwargs: any,
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

        max_retries = kwargs["max_retries"] if "max_retries" in kwargs else MAX_RETRIES
        retry_delay = kwargs["retry_delay"] if "retry_delay" in kwargs else RETRY_DELAY

        for attempt in range(max_retries):
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
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("All retry attempts failed.")
                    raise

    @property
    def _identifying_params(self) -> dict[str, any]:
        return {
            "model_name": self.model,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty
        }

    @property
    def _llm_type(self) -> str:
        return "openrouter_llm"