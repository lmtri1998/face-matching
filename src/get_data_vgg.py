import sys
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2
import os
from sklearn.preprocessing import normalize
# Algorithm ACE
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
# VGG_Face
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

ap = argparse.ArgumentParser()

ap.add_argument("--dataset", default="../datasets/",
                help="Path to training dataset")
ap.add_argument("--data", default="data_vgg.pickle")

# Argument of insightface
ap.add_argument('--image-size', default='224,224', help='')
ap.add_argument('--folder', default='matching_out/')

args = ap.parse_args()
folder__ = args.folder
if not (os.path.exists(folder__)):
    os.mkdir(folder__)
# Grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
#imagePaths = list(paths.list_images(args.dataset))

# Initialize the faces embedder with fc6 of VGG16
hidden_dim = 4096
vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
out = Dense(hidden_dim, activation='relu', name='fc6')(x)
embedding_model = Model(vgg_model.input, out)

# Initialize our lists of extracted facial embeddings and corresponding people names
embeddings = []
labels = []

#embeddings_non_norm = []

# Normalization vector diffrence
def norm_vec(vec_id, vec_selfie, p_norm='l2'):
    norm = normalize([vec_id, vec_selfie], norm='l2')
    normalized = np.abs(norm[0] - norm[1])
    return normalized

# Loop over the imagePaths
for label in os.listdir(args.dataset):
    count = 0
    label_folder = args.dataset + label + '/'
    for i,sub_folder in enumerate(os.listdir(label_folder)):
        image_path = label_folder + sub_folder + '/'
        vec_id = None
        embeddings_selfie = []
        for path in os.listdir(image_path):
            file_path = image_path + path
            if (path.find("_id") > -1):
                count += 1
            if('.directory' in file_path):
                continue
            # extract the person name from the image path
            print("[INFO] Label {} processing folder {}/{}".format(label,i+1, len(os.listdir(label_folder))))
            # load the image
            image = cv2.imread(file_path)
            # convert face to RGB color
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    # ACE
            #nimg = cca.automatic_color_equalization(nimg)
	    # Resize image from 112,112 to 224,224
            nimg = cv2.resize(nimg, (224,224))

	    # convert into an array of samples
            samples = np.asarray([nimg], 'float32')
            samples = preprocess_input(samples, version=1)

            # Get the face embedding vector => output vector have dim 4096
            face_embedding = embedding_model.predict(samples)
            face_embedding = face_embedding.reshape(4096,)
            #print(face_embedding.shape)

            # embedding to their respective list
            if (path.find("_id") > -1):
                vec_id = face_embedding
            else:
                embeddings_selfie.append(face_embedding)
        # Sub two vector
        for vec_selfie in embeddings_selfie:
            embeddings.append(norm_vec(vec_id, vec_selfie))
            #embeddings_non_norm.append(vec_id - vec_selfie)
            labels.append(label)
    print(count)
# save to output
data = {"data": embeddings, "labels": labels}
f = open(folder__ + args.data, "wb")
f.write(pickle.dumps(data))
f.close()

#data = {"data": embeddings_non_norm, "labels": labels}
#f = open(folder__ + "data_vgg_non_norm.pickle", "wb")
#f.write(pickle.dumps(data))
#f.close()
