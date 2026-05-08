import json
import os

from datasets import DatasetDict, load_dataset
from huggingface_hub import login


def add_image_tag(example):
    question = example.get('question', '')
    img = example.get('image', None)
    return {
        'problem': f"<image>{question}",
        'answer': example.get('answer', ''),
        'images': [img] if img is not None else [],
    }


def _login_hf():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token is None and os.path.exists("tokens.json"):
        with open("tokens.json", "r") as f:
            token = json.load(f).get("huggingface")
    if token is None:
        raise RuntimeError("Set HF_TOKEN or create tokens.json from tokens.json.example")
    login(token=token)


def main():
    HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
    if HUGGINGFACENAME is None:
        raise RuntimeError("HUGGINGFACENAME env var is not set")
    _login_hf()

    print("Loading FanqingM/MMK12 ...")
    dataset = load_dataset("FanqingM/MMK12")
    print(f"  train: {len(dataset['train'])}")
    print(f"  test:  {len(dataset['test'])}")

    train_processed = dataset['train'].map(add_image_tag)
    test_processed = dataset['test'].map(add_image_tag)

    processed = DatasetDict({"train": train_processed, "test": test_processed})

    repo_id = f"{HUGGINGFACENAME}/MMK12"
    print(f"Uploading to {repo_id} ...")
    processed.push_to_hub(repo_id, private=True)
    print("Done.")


if __name__ == "__main__":
    main()
