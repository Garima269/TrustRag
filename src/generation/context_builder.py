from typing import List, Dict

class ContextBuilder:

    def __init__(self, max_context_chars: int = 12000):
        self.max_context_chars = max_context_chars

    def build_context(self, documents: List[Dict]) -> str:

        if not documents:
            return ""

        context_parts = []

        for document in documents:

            title = document.get("title", "Unknown")
            text = document.get("text", "")

            if not text:
                continue

            context_parts.append(f"Title: {title}\n" f"Content: {text}")

        context = "\n\n".join(context_parts)

        # Keep context within a manageable size.
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars]

        return context


if __name__ == "__main__":

    documents = [
        {"title": "Example Document", "text": "Delhi is the capital of India."},
        {"title": "Another Document", "text": "India is located in South Asia."},
    ]

    builder = ContextBuilder()

    context = builder.build_context(documents)

    print("\nGenerated Context:\n")
    print(context)
