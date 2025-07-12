import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ActivationsExtractor:
    """
    Handles loading the model locally to extract internal hidden states (activations).
    """
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.step_end_str = "</step>"
        # Ensure the token is a single token
        encoded_step = self.tokenizer.encode(self.step_end_str, add_special_tokens=False)
        if len(encoded_step) != 1:
            raise ValueError(f"The string '{self.step_end_str}' is not a single token, it's {len(encoded_step)} tokens: {encoded_step}")
        self.step_end_token_id = encoded_step[0]
        print(f"✅ ModelActivationExtractor initialized on device: {self.device}")

    def get_step_activations(self, prompt_text):
        """
        Performs a forward pass and returns the hidden state of the last </step> token.

        Args:
            prompt_text (str): The full input text, which should contain </step>.

        Returns:
            torch.Tensor or None: The activations of the last </step> token, or None if not found.
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]

        # Find all occurrences of the step_end_token_id
        step_token_indices = torch.where(input_ids == self.step_end_token_id)[0]

        if len(step_token_indices) == 0:
            print(f"⚠️ Warning: '{self.step_end_str}' token ID ({self.step_end_token_id}) not found in prompt.")
            return None

        # Get the index of the last occurrence
        last_step_token_index = step_token_indices[-1]

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get the hidden states from the last layer
        last_hidden_states = outputs.hidden_states[-1]
        
        # Get the activations for the very last token in the sequence
        step_token_activations = last_hidden_states[0, last_step_token_index, :]
        return step_token_activations.cpu() # Move to CPU for storage