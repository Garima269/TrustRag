from src.utils.config import (
    MODEL_NAME,
    MAX_NEW_TOKENS,
)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class AnswerGenerator:

    def __init__(self):

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Loading model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )

        self.model.eval()

        print("Model loaded successfully!")

    def build_prompt(self, question: str, context: str) -> str:
    
        prompt = f"""You are a factual question-answering assistant.

    Use ONLY the information provided in the retrieved context.

    Rules:
    1. Return ONLY the shortest possible answer.
    2. Do NOT explain your reasoning.
    3. Do NOT write complete sentences.
    4. Do NOT repeat the question.
    5. If the answer is a person's name, return only the name.
    6. If the answer is a date, return only the date.
    7. If the answer is a place, return only the place name.
    8. If the answer is not present in the context, reply exactly:
    "I don't have enough information in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

        return prompt

    def generate_answer(self, question: str, context: str) -> str:

        prompt = self.build_prompt(question, context)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Only decode newly generated tokens.
        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[0, input_length:]

        answer = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        return answer


if __name__ == "__main__":

    context = """
Delhi is the capital of India.
India is located in South Asia.
"""

    question = "What is the capital of India?"

    generator = AnswerGenerator()

    answer = generator.generate_answer(question, context)

    print("\nGenerated Answer:\n")
    print(answer)
