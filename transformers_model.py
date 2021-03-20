import transformers
import datasets
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

print(dataset_train.column_names)



P, R, F1 = score(cands, refs, lang='en', verbose=True)
print(tf.math.reduce_mean(P, axis=None, keepdims=False, name=None))
print(tf.math.reduce_mean(R, axis=None, keepdims=False, name=None))
print(tf.math.reduce_mean(F1, axis=None, keepdims=False, name=None))

plot_example(cands[0],refs[0], lang="en")
plot_example(cands[0],refs[0], lang="en", rescale_with_baseline=True)