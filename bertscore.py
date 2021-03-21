
import bert_score
print(bert_score.__version__)
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["xtick.major.size"] = 0
rcParams["xtick.minor.size"] = 0
rcParams["ytick.major.size"] = 0
rcParams["ytick.minor.size"] = 0

rcParams["axes.labelsize"] = "large"
rcParams["axes.axisbelow"] = True
rcParams["axes.grid"] = True

from bert_score import score

# with open("distilbart-cnn-12-3.txt") as f:
#     model = "BART"
# with open("bart_large2.txt") as f:
#     model = "RoBERTa"
# with open("hyps.txt") as f:
#     model = "PEGASUS-CNN"
# with open("pegasus_reddit_short.txt") as f:
#     model = "PEGASUS-reddit""

with open("distilbart_cnn_long.txt") as f:
    model = "BART"
# with open("pegasus_cnn_long.txt") as f:
#     model = "PEGASUS-CNN"
# with open("pegasus_reddit_long.txt") as f:
#     model = "PEGASUS-reddit"
# with open("roberta_cnn_long.txt") as f:
#     model = "RoBERTa"

    cands = [line.strip() for line in f]

# with open("refs_short.txt") as f:
#     data = "short"

with open("refs_long.txt") as f:
    data = "long"

    refs = [line.strip() for line in f]

print("{} on reddit {}".format(model, data))
cands = cands[:len(refs)]
refs = refs[:len(cands)]

P, R, F1 = score(cands, refs, lang='en', verbose=True, rescale_with_baseline=True)
print([i for i in F1])
print(f"System level F1 mean: {F1.mean():.3f}")
print(f"System level F1 standard deviation: {F1.std():.3f}")
print(f"System level F1 max value: {F1.max():.3f}")


plt.hist(F1, bins=20)
plt.xlabel("score")
plt.ylabel("counts")
plt.show()