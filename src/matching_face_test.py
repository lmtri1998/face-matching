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


ap = argparse.ArgumentParser()

ap.add_argument("--models", default="fold_1_svm.pickle",
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

def face_embedding(image_path):
    # Return embedding
    read = cv2.imread(image_path)
    ret = detector.detect(read.copy(), 0.5, do_flip=False)
    bbox, landmarkss = ret
    if(len(bbox) == 0):
        return bbox
    embedding =  extract_feature(read.copy(), bbox[0],landmarkss[0])
    return embedding

def predict_matching(id_image, selfie_image):
    embedding_id = face_embedding(id_image)
    embedding_selfie = face_embedding(selfie_image)
    if(len(embedding_id) == 0 or len(embedding_selfie) == 0):
        return -1
    distain_vec = norm_vec(embedding_id, embedding_selfie)

    #predict
    result = model.predict([distain_vec])
    return result[0]

# TEST
match_path = glob('../testsets/matching/*/')
no_match_path = glob('../testsets/no_matching/*/')
matching_count = 0
no_match_count = 0
no_face_found_list = []
wrong_match_list = []
wrong_no_match_list = []
total = 30
pattern = re.compile("^[0-9]+\.jpg")

# test matching
for sd in match_path:
    selfie_img = ""
    for f in os.listdir(sd):
        if pattern.match(f):
            selfie_img = os.path.join(sd, f)
    print(selfie_img)
    for file in glob(os.path.join(sd,"cccd_*.jpg")):
        print(file)
        res = predict_matching(file, selfie_img)
        if res == 1:
            print('Matching')
            matching_count += 1
        elif res == -1:
            no_face_found_list.append(sd)
            print('No Face found')
        else:
            wrong_match_list.append(sd)
            print('No matching')        

    for file in glob(os.path.join(sd,"cmnd_*.jpg")):
        print(file)
        res = predict_matching(file, selfie_img)
        if res == 1:
            print('Matching')
            matching_count += 1
        elif res == -1:
            no_face_found_list.append(sd)
            print('No Face found')
        else:
            wrong_match_list.append(sd)
            print('No matching')    

# test no matching
# for sd in no_match_path:
#     selfie_img = ""
#     for f in os.listdir(sd):
#         if pattern.match(f):
#             selfie_img = os.path.join(sd, f)
#     print(selfie_img)
#     for file in glob(os.path.join(sd,"cccd_*.jpg")):
#         print(file)
#         res = predict_matching(file, selfie_img)
#         if res == 1:
#             print('Matching')
#             wrong_no_match_list.append(sd)
#         elif res == -1:
#             no_face_found_list.append(sd)
#             print('No Face found')
#         else:
#             no_match_count += 1
#             print('No matching')        

#     for file in glob(os.path.join(sd,"cmnd_*.jpg")):
#         print(file)
#         res = predict_matching(file, selfie_img)
#         if res == 1:
#             print('Matching')
#             wrong_no_match_list.append(sd)
#         elif res == -1:
#             no_face_found_list.append(sd)
#             print('No Face found')
#         else:
#             no_match_count += 1
#             print('No matching')        

print("Matching Correct " + str(matching_count/total) + '\n')
# print("No Matching Correct " + str(no_matching_count/total) + '\n')
print("No Face Found", no_face_found_list)
print("Wrong Matching", wrong_match_list)
# print("Wrong No Matching", wrong_no_match_list)
# res = predict_matching(args.id_image, args.selfie_image)
# print(res)