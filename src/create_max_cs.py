import numpy as np
import argparse
import pickle
from sklearn.preprocessing import normalize
# Construct the argumet parser and parse the argument
ap = argparse.ArgumentParser()

ap.add_argument("--embeddings", default="../embeddings.pickle",
                help="path to embedding")
ap.add_argument("--folder", default="./matching_out_2865_max/")

ap.add_argument("--out", default="data_insight.pickle")

args = vars(ap.parse_args())

data = pickle.loads(open(args["embeddings"], "rb").read())
labels = data["labels"]
data = data["data"]

def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

# Normalization vector diffrence
def norm_vec(vec_id, p_norm='l2'):
    norm = normalize([vec_id], norm='l2')
    return norm[0]

# Normalization vector diffrence
def sub_vec(vec_id, vec_selfie, p_norm='l2'):
    norm = normalize([vec_id, vec_selfie], norm='l2')
    normalized = np.abs(norm[0] - norm[1])
    return normalized

def find_id_max(embedding, index):
    max_cs = 0
    index_max = -1
    for ix_labels,label in enumerate(labels):
        if '_id' in label and ix_labels != index:
            cosine = CosineSimilarity(norm_vec(embedding), [norm_vec(data[ix_labels])])
            if cosine > max_cs:
                max_cs = cosine
                index_max = ix_labels
    return labels[index_max].split("_id")[0] + "_selfie",index_max

X = []
y = []
count1 = 0
count0 = 0
# Create data
for ix_labels,label in enumerate(labels):
    if '_id' in label:
        print("-----------------{}-------------------".format(ix_labels))
        selfie = label.split("_id")[0] + "_selfie"
        import time
        selfie_max_cs, index_max = find_id_max(data[ix_labels], ix_labels)
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
        for ix in range(index_max - 20, index_max + 20):
            if ix >= len(labels) or ix < 0:
                continue
            label_max = labels[ix]
            # Data selfie của ID có khoảng các MAX
            if selfie_max_cs == label_max:
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
