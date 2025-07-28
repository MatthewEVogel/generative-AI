from datasets import load_dataset
import unicodedata
import string

# Stream a dataset like OpenWebText2 or Wikipedia
dataset = load_dataset("EleutherAI/pile", streaming=True, split="train")


def normalize_unicode(text):
    return unicodedata.normalize("NFKC", text)


def remove_non_printable(text):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, text))


def normalize_whitespace(text):
    return ' '.join(text.split())


def is_valid(text):
    return len(text.strip()) > 10  # Must be > 10 characters


def preprocess(text):
    text = normalize_unicode(text)
    text = remove_non_printable(text)
    text = normalize_whitespace(text)
    return text


for example in dataset:
    raw_text = example["text"]
    processed = preprocess(raw_text)
    if is_valid(processed):
        # Use this for BPE training
        pass
