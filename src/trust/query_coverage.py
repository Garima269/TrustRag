import re

class QueryCoverage:
    """
    Computes query coverage between
    a question and a retrieved sentence.
    """

    def __init__(self):

        print("Query Coverage Module Ready!\n")

    def preprocess(self, text: str):

        text = text.lower()

        words = re.findall(r"\b\w+\b", text)

        return set(words)

    def compute_score(self, question: str, sentence: str):

        query_words = self.preprocess(question)

        sentence_words = self.preprocess(sentence)

        if len(query_words) == 0:
            return 0.0

        matched_words = query_words.intersection(sentence_words)

        score = len(matched_words) / len(query_words)

        return round(score, 4)


if __name__ == "__main__":

    question = "Who invented the electric battery?"

    sentence = "Alessandro Volta invented the electric battery."

    coverage = QueryCoverage()

    score = coverage.compute_score(question, sentence)

    print("Query Coverage Score\n")

    print(score)
