import os
import re
import requests
from openai import OpenAI
from transformers import AutoTokenizer


class BaseInference:
    """
    Base class for inference wrappers.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("jacopo-minniti/Qwen2.5-7B-base")
        self.think_end_str = "</think>"
        self.step_end_str = "</step>"
        print(f"✅ {self.__class__.__name__} initialized.")

    def _generate_full_prompt(self, problem, previous_steps=""):
        # Manually format the prompt according to Qwen's template for completions
        system_prompt = "You are a helpful assistant. Please solve the following math problem by thinking step-by-step enclosed in <think> tags. Do not use <step> tags, just newlines between steps."
        
        cot = "<think>"
        if previous_steps:
            cot += previous_steps
        
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n{cot}"
        )

        return prompt

    def _convert_to_discrete_cot(self, continuous_cot):
        """Converts a newline-separated CoT to a discrete <step>-based one."""
        steps = continuous_cot.strip().split('\n\n')
        discrete_cot = "<think>\n"
        for step in steps:
            if step:
                discrete_cot += f"<step>\n{step}\n</step>\n"
        discrete_cot += "</think>"
        return discrete_cot

    def generate_from_step(self, problem, current_step_prompt, max_thinking_tokens, max_answer_tokens, n, temperature=1):
        raise NotImplementedError("This method should be implemented by subclasses.")


class APIInference(BaseInference):
    """
    A wrapper to handle text generation using the OpenRouter API.
    """
    def __init__(self, model_name):
        super().__init__(model_name)
        # It's recommended to use environment variables for API keys
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
            
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_from_step(self, problem, current_step_prompt, max_thinking_tokens, max_answer_tokens, n, temperature=1):
        """
        Takes the prompt up to a certain step and generates a full CoT and a final answer.
        This implementation uses a single generation call to the completions API.
        """

        full_prompt = self._generate_full_prompt(problem, current_step_prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use the completions endpoint
        self.url = "https://openrouter.ai/api/v1/completions"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "temperature": temperature,
            # "max_tokens": max_thinking_tokens+max_answer_tokens,
            "reasoning": {
                # "max_tokens": max_thinking_tokens
                "effort": "low"
            }
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()
            final_answer_part = completion['choices'][0].get('text')
            thinking_content = completion['choices'][0].get('reasoning')

            if final_answer_part is None or final_answer_part.strip() == "":
                print(f"⚠️ Error: API returned no content in the final answer ('text' field).\n Reasoning: {thinking_content}")
                return None

            if not thinking_content:
                print("⚠️ Warning: No 'reasoning' field found in API response.")

            # Extract final answer from the 'text' part
            boxed_answer = re.search(r"\\boxed{([^}]+)}", final_answer_part)
            final_answer = boxed_answer.group(1) if boxed_answer else final_answer_part.strip()
            return final_answer

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error during API inference: {e}")
            if e.response:
                print(f"Response body: {e.response.text}")
            return None
        except Exception as e:
            print(f"⚠️ Error during API inference: {e}")
            return None


class VLLMInference(BaseInference):
    """
    A wrapper to handle fast text generation using a vLLM server.
    """
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="key")

    def generate_from_step(self, problem, current_step_prompt, max_thinking_tokens, max_answer_tokens, n, temperature=1):
        """
        Takes the prompt up to a certain step and generates a full CoT and a final answer.
        This implementation uses a single generation call to the completions API.
        It can generate `n` completions in a single call.
        """
        full_prompt = self._generate_full_prompt(problem, current_step_prompt)

        try:
            # Note: vLLM's OpenAI endpoint can take a list of prompts, but we pass one and use n.
            # If different temperatures per generation are needed, multiple API calls would be required
            # as the standard completions endpoint doesn't support per-request sampling params in a batch.
            completion = self.client.completions.create(
                model=self.model_name,
                prompt=full_prompt,
                max_tokens=max_thinking_tokens + max_answer_tokens,
                stream=False,
                temperature=temperature,
                stop=self.think_end_str,
                n=n, # Generate n completions
                extra_body={'skip_special_tokens': False}
            )
            
            final_answers = []
            for choice in completion.choices:
                generated_text = choice.text
                
                # The model should generate both reasoning and the final answer.
                # We split the thinking part from the final answer part.
                if self.think_end_str in generated_text:
                    thinking_content, final_answer_part = generated_text.split(self.think_end_str, 1)
                else:
                    # If the model didn't output </think>, we assume the whole output is reasoning.
                    # We append </think> and generate the final answer in a second step.
                    thinking_content = generated_text
                    
                    # Construct prompt for the second generation to get the final answer
                    answer_prompt = full_prompt + thinking_content + f"\n{self.think_end_str}\nSo final answer is: "
                    
                    answer_completion = self.client.completions.create(
                        model=self.model_name,
                        prompt=answer_prompt,
                        max_tokens=max_answer_tokens,
                        stream=False,
                        temperature=temperature,
                        stop="<|im_end|>", # Stop at the end of the assistant's turn
                        extra_body={'skip_special_tokens': False}
                    )
                    final_answer_part = answer_completion.choices[0].text

                if not thinking_content:
                    print("⚠️ Warning: No thinking content was generated.")

                # Extract final answer from the 'text' part
                boxed_answer = re.search(r"\\boxed{([^}]+)}", final_answer_part)
                final_answer = boxed_answer.group(1) if boxed_answer else final_answer_part.strip()

                if not final_answer:
                    print(f"⚠️ Error: API returned no content in the final answer part.\n Full generation: {generated_text}")
                    final_answers.append(None)
                else:
                    final_answers.append(final_answer)

            return final_answers

        except Exception as e:
            print(f"⚠️ Error during vLLM inference: {e}")
            return [None] * n