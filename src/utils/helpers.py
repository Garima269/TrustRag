import json
from pathlib import Path


def ensure_directory(path):
    """
    Create a directory if it does not already exist.
    """

    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    """
    Save Python data as a JSON file.
    """

    path = Path(path)

    ensure_directory(path.parent)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


def load_json(path):
    """
    Load data from a JSON file.
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)