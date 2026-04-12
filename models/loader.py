from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ModelLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()

    def _get_device(self):
        """
        Decide whether to use GPU or CPU
        """
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self):
        """
        Load model and tokenizer
        """
        print(f"Loading model: {self.model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

        print("Model loaded successfully")

        return self.model, self.tokenizer

    def get_device(self):
        return self.device