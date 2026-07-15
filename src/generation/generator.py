"""
generator.py

This module generates answers using a Hugging Face
instruction-tuned language model.

Input:
    - User Question
    - Retrieved Context

Output:
    - Generated Answer
"""

from src.utils.config import (
    MODEL_NAME,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DO_SAMPLE,
)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class AnswerGenerator:
    """
    Generate answers using an instruction-tuned LLM.
    """

    def __init__(self):
        """
        Load tokenizer and language model.
        """

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

        print("Model loaded successfully!")

    def build_prompt(self, question: str, context: str) -> str:
        """
        Build the prompt for the language model.
        """

        prompt = f"""
You are a factual Question Answering assistant.

Instructions:
1. Answer ONLY using the provided context.
2. Keep the answer short and precise.
3. Do NOT add extra information.
4. If the answer is not present in the context, reply exactly:
"I don't have enough information in the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

        return prompt

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using the language model.
        """

        prompt = self.build_prompt(question, context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

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