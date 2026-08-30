import re
from typing import List, Dict

class TextCleaner:

    def clean_sentence(self, sentence_record: Dict) -> Dict:
        """
        Clean a single sentence.

        Parameters:
            sentence_record (dict)

        Returns:
            dict
        """

        cleaned_sentence = sentence_record.get("sentence_text", "")
        cleaned_sentence = str(cleaned_sentence)

        # Remove newline characters
        cleaned_sentence = cleaned_sentence.replace("\n", " ")

        # Remove tab characters
        cleaned_sentence = cleaned_sentence.replace("\t", " ")

        # Remove multiple spaces
        cleaned_sentence = re.sub(r"\s+", " ", cleaned_sentence)

        # Remove leading/trailing spaces
        cleaned_sentence = cleaned_sentence.strip()

        # Create a copy to avoid modifying the original dictionary

        # Preserve original metadata while updating only the cleaned sentence
        cleaned_record = sentence_record.copy()
        cleaned_record["sentence_text"] = cleaned_sentence

        return cleaned_record

    def clean_sentences(self, sentence_records: List[Dict]) -> List[Dict]:

        cleaned_sentences = []

        for sentence_record in sentence_records:
            cleaned_sentences.append(
                self.clean_sentence(sentence_record)
            )

        return cleaned_sentences

if __name__ == "__main__":

    sample_sentences = [
        {
            "doc_id": 1,
            "title": "Delhi",
            "rank": 1,
            "retriever_score": 0.95,
            "sentence_id": 1,
            "sentence_text": "   Delhi is the capital of India.\n\n"
        },
        {
            "doc_id": 2,
            "title": "Mumbai",
            "rank": 2,
            "retriever_score": 0.91,
            "sentence_id": 1,
            "sentence_text": "\tMumbai   is India's financial capital.   "
        }
    ]

    cleaner = TextCleaner()

    cleaned_sentences = cleaner.clean_sentences(sample_sentences)

    print("\nCleaned Sentences:\n")

    for sentence in cleaned_sentences:
        print(sentence)