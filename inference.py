import json
import requests
import datasets

split_patterns = {
        "train": "train[:5%]"
    }

dataset_train = datasets.load_dataset('reddit_tifu', 'long', split=split_patterns["train"])

# body: ['documents']
# title: ['title']


headers = {"Authorization": f"Bearer api_QniHzMaGMvtvUpernwhmCFFeegSRUFUNNj"}
API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

data = []

with open("distilbart_cnn_long.txt", 'a') as outfile:
    for i in range(len(dataset_train)):
        print("Generating entry ",i," out of ",len(dataset_train))
        data.append(query(
            {
                "inputs": dataset_train["documents"][i],
                "options": {"wait_for_model": True}
            }
        ))
        print(data[-1])
    json.dump(data, outfile)

