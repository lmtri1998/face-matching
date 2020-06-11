import numpy as np
import random
import argparse
import pickle
from sklearn.preprocessing import normalize
# Construct the argumet parser and parse the argument
ap = argparse.ArgumentParser()

ap.add_argument("--embeddings", default="../embeddings.pickle",
                help="path to embedding")
ap.add_argument("--folder", default="./matching_out_2865_namnv78/")

ap.add_argument("--out", default="data_insight.pickle")

ap.add_argument("--kind_pos", default="max")

ap.add_argument("--kind_neg", default="max")

args = vars(ap.parse_args())

out = "POS_" + args['kind_pos'] + "_Neg_" + args['kind_neg'] + ".pickle"

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

# Tìm ra index của embeddings selfie xa nhất
def find_selfie_max(embedding, index, kind='pos'):
    label_id = labels[index].split("_id")[0]
    max_cs = 0
    index_max = -1
    if kind == 'pos':
        for ix_labels in range(index - 20, index + 20):
            if ix_labels >= len(labels) or ix_labels < 0:
                continue
            labels_max = labels[ix_labels].split("_selfie")[0]
            if '_selfie' in labels[ix_labels] and label_id == labels_max:
                cosine = CosineSimilarity(norm_vec(embedding), [norm_vec(data[ix_labels])])
                if cosine > max_cs:
                    max_cs = cosine
                    index_max = ix_labels
    elif kind == 'neg':
        for ix_labels,label in enumerate(labels):
            labels_max = labels[ix_labels].split("_selfie")[0]
            if '_selfie' in label and label_id != labels_max:
                cosine = CosineSimilarity(norm_vec(embedding), [norm_vec(data[ix_labels])])
                if cosine > max_cs:
                    max_cs = cosine
                    index_max = ix_labels
    return index_max

# Tìm ra index của embeddings selfie gần nhất
def find_selfie_min(embedding, index, kind='pos'):
    label_id = labels[index].split("_id")[0]

    min_cs = 2
    index_min = -1
    if kind == 'pos':
        for ix_labels in range(index - 20, index + 20):
            if ix_labels >= len(labels) or ix_labels < 0:
                continue
            labels_min = labels[ix_labels].split("_selfie")[0]
            if '_selfie' in labels[ix_labels] and label_id == labels_min:
                cosine = CosineSimilarity(norm_vec(embedding), [norm_vec(data[ix_labels])])
                if cosine < min_cs:
                    min_cs = cosine
                    index_min = ix_labels
    elif kind == 'neg':
        for ix_labels,label in enumerate(labels):
            labels_min = labels[ix_labels].split("_selfie")[0]
            if '_selfie' in label and label_id != labels_min:
                cosine = CosineSimilarity(norm_vec(embedding), [norm_vec(data[ix_labels])])
                if cosine < min_cs:
                    min_cs = cosine
                    index_min = ix_labels
    return index_min

# Tìm ra index của embeddings selfie ngẫu nhiên
def find_selfie_random(index, kind='pos'):
    label_id = labels[index].split("_id")[0]
    if kind == 'pos':
        labels_random = labels[index].split("_selfie")[0]
        while (not(label_id == labels_random)):
            index_random = random.randint(index - 20, index + 20)
            if index_random >= len(labels) or index_random < 0:
                continue
            labels_random = labels[index_random].split("_selfie")[0]
    elif kind == 'neg':
        index_random = random.randint(0, len(labels) - 1)
        labels_random = labels[index_random].split("_selfie")[0]
        while(not('_selfie' in labels[index_random] and label_id != labels_random)):
            index_random = random.randint(0, len(labels) - 1)
            labels_random = labels[index_random].split("_selfie")[0]
    return index_random

X = []
y = []
count1 = 0
count0 = 0
# Create data
for ix_labels,label in enumerate(labels):
    if '_id' in label:
        print("-----------------{}-------------------".format(ix_labels))
        selfie = label.split("_id")[0] + "_selfie"
        
        if args['kind_pos'] == "max":
            index_max = find_selfie_max(data[ix_labels], ix_labels, 'pos')
            X.append(sub_vec(data[ix_labels], data[index_max]))
            y.append(1)
            count1 += 1
        elif args['kind_pos'] == "min":
            index_min = find_selfie_min(data[ix_labels], ix_labels, 'pos')
            X.append(sub_vec(data[ix_labels], data[index_min]))
            y.append(1)
            count1 += 1
        elif args['kind_pos'] == "random":
            index_random = find_selfie_random(ix_labels, 'pos')
            X.append(sub_vec(data[ix_labels], data[index_random]))
            y.append(1)
            count1 += 1
        else:
            raise("ERROR argument")

        if args['kind_neg'] == "max":
            index_max = find_selfie_max(data[ix_labels], ix_labels, 'neg')
            X.append(sub_vec(data[ix_labels], data[index_max]))
            y.append(0)
            count0 += 1
        elif args['kind_neg'] == "min":
            index_min = find_selfie_min(data[ix_labels], ix_labels, 'neg')
            X.append(sub_vec(data[ix_labels], data[index_min]))
            y.append(0)
            count0 += 1
        elif args['kind_neg'] == "random":
            index_random = find_selfie_random(ix_labels, 'neg')
            X.append(sub_vec(data[ix_labels], data[index_random]))
            y.append(0)
            count0 += 1
        else:
            raise("ERROR argument")

print(count0)
print(count1)
data = {"data": X, "labels": y}
f = open(args["folder"] + out, "wb")
f.write(pickle.dumps(data))
f.close()
