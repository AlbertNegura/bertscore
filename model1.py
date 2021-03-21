import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.summarization import reddit_tifu, reddit_tifu_test
from bert_score import score
from bert_score import plot_example
# hide the loading messages
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import rcParams
import tokenizer

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
cands, refs = ["Abstractive summarization is a natural language generation task where new phrases that capture the most salient information from an article, passage, or paragraph are generated with the objec-tive of shortening the length of the original document.  The task is described as “abstractive” to accentuate that, in contrast to the extractive summarization,  the generated text is a compressed paraphrasing of the main contents of the document, potentially using vocabulary unseen in the source document.", "I am a potato and I'm ok, I like to eat food"], ["Text summarization is the task of creating from along text document, a shorter, and coherent version in order to discover and consume relevant information faster.", "Potatoes are a vegetable that will take over the world."]



# check bert-score + plots
P, R, F1 = score(cands, refs, lang='en', verbose=True)
print(tf.math.reduce_mean(P, axis=None, keepdims=False, name=None))
print(tf.math.reduce_mean(R, axis=None, keepdims=False, name=None))
print(tf.math.reduce_mean(F1, axis=None, keepdims=False, name=None))

plot_example(cands[0],refs[0], lang="en")
plot_example(cands[0],refs[0], lang="en", rescale_with_baseline=True)