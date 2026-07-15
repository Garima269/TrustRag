"""
generator.py

This module generates answers using a Hugging Face
instruction-tuned language model.

Model:
    Qwen/Qwen2.5-1.5B-Instruct

Input:
    Question
    Context

Output:
    Generated Answer
"""
from src.utils.config import MODEL_NAME
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class AnswerGenerator:
    """
    Generates answers using the Qwen2.5-1.5B-Instruct model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    ):
        """
        Load tokenizer and language model.
        """

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

        print("Model loaded successfully!")

    def build_prompt(self, question: str, context: str) -> str:
        """
        Create the prompt for the language model.

        Parameters:
            question (str)
            context (str)

        Returns:
            str
        """

        prompt = f"""
You are a factual Question Answering assistant.

Use ONLY the information present in the context.

If the answer cannot be found in the context,
reply with:

"I don't have enough information in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

        return prompt

    def generate_answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 150
    ) -> str:
        """
        Generate an answer.

        Parameters:
            question (str)
            context (str)

        Returns:
            str
        """

        prompt = self.build_prompt(question, context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Keep only the generated answer
        answer = generated_text.split("Answer:")[-1].strip()

        return answer


if __name__ == "__main__":

    context = """
Delhi is the capital of India.

It is located in northern India.

India is a country in South Asia.
"""

    question = "What is the capital of India?"

    generator = AnswerGenerator()

    answer = generator.generate_answer(
        question,
        context
    )

    print("\nGenerated Answer:\n")
    print(answer)