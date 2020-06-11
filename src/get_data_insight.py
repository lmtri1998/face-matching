import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from imutils import paths
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import cv2
import os
from sklearn.preprocessing import normalize
# Algorithm ACE
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil

ap = argparse.ArgumentParser()

ap.add_argument("--dataset", default="../datasets_2865_max/",
                help="Path to training dataset")
ap.add_argument("--data", default="data_insight.pickle")

# Argument of insightface
ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-r100-ii/model,0', help='path to load model.')
ap.add_argument('--folder', default='matching_out_2865_max/')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()
folder__ = args.folder
if not (os.path.exists(folder__)):
    os.mkdir(folder__)
# Grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
#imagePaths = list(paths.list_images(args.dataset))

# Initialize the faces embedder
embedding_model = face_model.FaceModel(args)

# Initialize our lists of extracted facial embeddings and corresponding people names
embeddings = []
labels = []
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
            if (path.find("id") > -1):
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

            nimg = np.transpose(nimg, (2,0,1))
            # Get the face embedding vector
            face_embedding = embedding_model.get_feature(nimg)
            if (path.find("id") > -1):
                vec_id = face_embedding
            else:
                embeddings_selfie.append(face_embedding)
        # Sub two vector
        for vec_selfie in embeddings_selfie:
            embeddings.append(norm_vec(vec_id, vec_selfie))
            labels.append(label)
    print(count)
# save to output
data = {"data": embeddings, "labels": labels}
f = open(folder__ + args.data, "wb")
f.write(pickle.dumps(data))
f.close()
