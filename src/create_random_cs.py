import numpy as np
import random
import argparse
import pickle
from sklearn.preprocessing import normalize
# Construct the argumet parser and parse the argument
ap = argparse.ArgumentParser()

ap.add_argument("--embeddings", default="../embeddings.pickle",
                help="path to embedding")
ap.add_argument("--folder", default="./matching_out_2865_random/")

ap.add_argument("--out", default="data_insight.pickle")

args = vars(ap.parse_args())

data = pickle.loads(open(args["embeddings"], "rb").read())
labels = data["labels"]
data = data["data"]

# Normalization vector diffrence
def sub_vec(vec_id, vec_selfie, p_norm='l2'):
    norm = normalize([vec_id, vec_selfie], norm='l2')
    normalized = np.abs(norm[0] - norm[1])
    return normalized
X = []
y = []
count1 = 0
count0 = 0
# Create data
for ix_labels,label in enumerate(labels):
    if '_id' in label:
        print("-----------------{}-------------------".format(ix_labels))
        selfie = label.split("_id")[0] + "_selfie"
        index_random = random.randint(0, len(labels) - 1)
        while(not(index_random != ix_labels and "_id" in labels[index_random])):
            index_random = random.randint(0, len(labels) - 1)
        selfie_random_cs = labels[index_random].split("_id")[0] + "_selfie"
        print(1)
        for ix in range(ix_labels - 20, ix_labels + 20):
            if ix >= len(labels) or ix < 0:
                continue
            label_selfie = labels[ix]
            if '_id' in label_selfie:
                continue
            # Data selfie của chính nó
            if selfie == label_selfie:
                print(labels[ix_labels], labels[ix])
                X.append(sub_vec(data[ix_labels], data[ix]))
                y.append(1)
                count1 += 1
        print(0)
        for ix in range(index_random - 20, index_random + 20):
            if ix >= len(labels) or ix < 0:
                continue
            label_random = labels[ix]
            # Data selfie của ID có khoảng các min
            if selfie_random_cs == label_random:
                import time
                print(labels[ix_labels], labels[ix])
                X.append(sub_vec(data[ix_labels], data[ix]))
                y.append(0)
                count0 += 1

print(count0)
print(count1)
data = {"data": X, "labels": y}
f = open(args["folder"] + args["out"], "wb")
f.write(pickle.dumps(data))
f.close()
