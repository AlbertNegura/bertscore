import json
import requests
import datasets

split_patterns = {
        "train": "train",
        "validation": "train[80%:90%]",
        "test": "train[90%:]"
    }

dataset_train = datasets.load_dataset('reddit_tifu', 'short', split=split_patterns["train"])
dataset_val = datasets.load_dataset('reddit_tifu', 'short', split=split_patterns["validation"])
dataset_test = datasets.load_dataset('reddit_tifu', 'short', split=split_patterns["test"])

# body: ['documents']
# title: ['title']


headers = {"Authorization": f"Bearer {API_KEY}"}
API_URL = "https://api-inference.huggingface.co/google/roberta2roberta_L-24_cnn_daily_mail"

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

data = []

for i in range(len(dataset_train)):
    data.append(query(
        {
            "inputs": dataset_train["documents"][i],
        }
    ))


with open({OUTPUT_FILE}, 'w') as outfile:
    json.dump(data, outfile)