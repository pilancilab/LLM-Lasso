from dataclasses import dataclass, field

@dataclass
class LLMParams:
    temp: float = field(default=0.5, metadata={
        "help": "Temperature for randomness in responses."
    })
    top_p: float = field(default=0.9, metadata={
        "help": "Top-p sampling parameter."
    })
    repetition_penalty: float = field(default=0.9, metadata={
        "help": "Penalty for repeated tokens."
    })
    model_type: str = field(default="gpt-4o", metadata={
        "help": "Type of model to use",
        "choices": ["gpt-4o", "o1", "openrouter"]
    })
    model_name: str = field(default=None, metadata={
        "help": "Name of the model to use. For openrouter LLMs, defaults to \"meta-llama/Llama-3.1-8B-Instruct\""
    })

    def get_model_name(self):
        if self.model_name:
            return self.model_name
        if self.model_type == "openrouter":
            return "meta-llama/Llama-3.1-8B-Instruct"
        return self.model_type
