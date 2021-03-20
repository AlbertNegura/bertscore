import transformers
import datasets

split_patterns = {
        "train": "train[:80%]",
        "validation": "train[80%:90%]",
        "test": "train[90%:]"
    }

dataset_train = datasets.load_dataset('reddit_tifu', 'short', split=split_patterns["train"])
dataset_val = datasets.load_dataset('reddit_tifu', 'short', split=split_patterns["validation"])
dataset_test = datasets.load_dataset('reddit_tifu', 'short', split=split_patterns["test"])

# body: ['documents']
# title: ['title']

print(dataset_train)

