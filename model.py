import tensorflow_datasets as tfds
from bert_score import score
from bert_score import plot_example
# hide the loading messages
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
from matplotlib import rcParams

rcParams["xtick.major.size"] = 0
rcParams["xtick.minor.size"] = 0
rcParams["ytick.major.size"] = 0
rcParams["ytick.minor.size"] = 0

rcParams["axes.labelsize"] = "large"
rcParams["axes.axisbelow"] = True
rcParams["axes.grid"] = True

split_patterns = {
        "train": "train[:90%]",
        "validation": "train[90%:]",
        "test": "test"
    }

def load(split):
    dataset, info = tfds.load(
        "reddit_tifu",
        split=split_patterns[split],
        with_info=True)
    if split == "train":
        num_examples = info.splits["train"].num_examples * 0.8
    elif split == "validation":
        num_examples = info.splits["train"].num_examples * 0.1
    else:
        num_examples = info.splits["train"].num_examples * 0.1
    return dataset, num_examples



# load data
data, examples = load("train")


# define model


# train model


# produce output
cands, refs = ["something"], ["something_else"]



# check bert-score + plots
P, R, F1 = score(cands, refs, lang='en', verbose=True)
print(P)
print(R)
print(F1)
plot_example(cands[0],refs[0], lang="en")
plot_example(cands[0],refs[0], lang="en", rescale_with_baseline=True)