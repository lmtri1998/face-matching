import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')
sys.path.append('../insightface/retinaface')

import random
from keras.models import load_model
from retinaface import RetinaFace
from sklearn.preprocessing import normalize
from imutils import paths
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import cv2
import os
from glob import glob
import re
import time

ap = argparse.ArgumentParser()

ap.add_argument("--models", default="fold_4_nn.pickle",
    help='Path to model')
ap.add_argument("--folder", default="matching_out_2865_max/",
    help='Folder store')
ap.add_argument("--id_image", default="../id1.jpg",
    help='Input id_image')
ap.add_argument("--selfie_image", default="../selfie1.jpg",
    help='Input selfie image')

ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-r100-ii/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

detector = RetinaFace("../insightface/retinaface/model/R50", 0, -1, 'net3')

# Initialize faces embedding model
embedding_model =face_model.FaceModel(args)

with open(args.folder + args.models, 'rb') as f:
    model = pickle.load(f)
#model._make_predict_function()

#------------------------------------
#if not os.path.exists('/static/rec/'):
#    os.makedirs('/static/rec/')

def extract_feature(path_img,bbox,landmark):
    # load the image
    image = cv2.resize(path_img,(112,112))
    nimg = face_preprocess.preprocess(image, bbox, landmark.astype(int), image_size='112,112')
    # convert face to RGB color
    nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nimg = np.transpose(nimg, (2,0,1))
    # Get the face embedding vector
    face_embedding = embedding_model.get_feature(nimg)
    return face_embedding

def norm_vec(vec_id, vec_selfie, p_norm='l2'):
    norm = normalize([vec_id, vec_selfie], norm='l2')
    normalized = np.abs(norm[0] - norm[1])
    return normalized

def get_bbox_and_landmarks(image):
    ret = detector.detect(image.copy(), 0.5, do_flip=False)
    return ret
    
def face_embedding(image, bbox, landmarkss):
    # Return embedding
    embedding =  extract_feature(image.copy(), bbox[0],landmarkss[0])
    return embedding

def predict_matching(id_image, selfie_image):
    start_time = time.time()
    id_bbox, id_landmark = get_bbox_and_landmarks(id_image)
    selfie_bbox, selfie_landmark = get_bbox_and_landmarks(selfie_image)
    embedding_id = face_embedding(id_image, id_bbox, id_landmark)
    embedding_selfie = face_embedding(selfie_image,  selfie_bbox, selfie_landmark)
    distain_vec = norm_vec(embedding_id, embedding_selfie)

    # predict
    result = model.predict_proba([distain_vec])
    total_time = time.time() - start_time
    return result[0], id_bbox, selfie_bbox, embedding_id, embedding_selfie, total_time

# id_img = cv2.imread(args.id_image)
# selfie_img = cv2.imread(args.selfie_image)
# res = predict_matching(id_img, selfie_img)
# print(res)