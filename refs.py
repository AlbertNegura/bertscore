import json
import requests
import datasets

split_patterns = {
        "train": "train[:5%]"
    }

dataset_train = datasets.load_dataset('reddit_tifu', 'short', split=split_patterns["train"])

# body: ['documents']
# title: ['title']


data = [i for i in dataset_train["title"]]
print(data)

with open("refs.txt", 'a') as outfile:
    json.dump(data, outfile)

