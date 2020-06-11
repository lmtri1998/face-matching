import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')
sys.path.append('../insightface/retinaface')

import random
from keras.models import load_model
from retinaface import RetinaFace
from imutils import paths
import face_preprocess
import numpy as np
import face_model
import argparse
import pickle
import cv2
import os

ap = argparse.ArgumentParser()

ap.add_argument("--models", default="fold_5_svm.pickle",
    help='Path to model')
ap.add_argument("--folder", default="matching_out_2865_max/",
    help='Folder store')
ap.add_argument("--id_image", default="../id.jpg",
    help='Input id_image')
ap.add_argument("--live_image", default="../selfie.jpg",
    help='Input live image')

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
    image = cv2.imread(path_img)
    image = cv2.resize(image,(112,112))
    nimg = face_preprocess.preprocess(image, bbox, landmark.astype(int), image_size='112,112')
    # convert face to RGB color
    nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    nimg = np.transpose(nimg, (2,0,1))
    # Get the face embedding vector
    face_embedding = embedding_model.get_feature(nimg)
    return face_embedding

def sub_two_vector(path_id, path_live, bbox_id, landmark_id, bbox_live, landmark_live):
    id_embedding = extract_feature(path_id, bbox_id, landmark_id)
    live_embedding = extract_feature(path_live, bbox_live, landmark_live)
    distance = live_embedding - id_embedding
    return distance

def save_crop_image(path_image):
    read = cv2.imread(path_image)
    print(5555555)
    print(path_image)
    ret = detector.detect(read, 0.5, do_flip=False)
    print("cccccccccccccccc")
    bbox, landmarkss = ret
    print(7777777)
    #print("dasdadasdasdads : ", bbox)
    name_crop = "static/rec/" + str(path_image).split("/")[-1][:-4].replace('.','') + ".png"
    cv2.imwrite(name_crop, read[int(bbox[0][1]): int(bbox[0][3]),int(bbox[0][0]):int(bbox[0][2])])
    return name_crop, bbox[0].astype(int), landmarkss[0].astype(int)

#------------------------------------
def predict_matching(id_img, live_img):
    #print(id_img)
    #print(live_img)
    print(11111111)
    id_crop,bbox_id,landmark_id = save_crop_image(id_img)
    live_crop,bbox_live,landmark_live = save_crop_image(live_img)
    print(22222222)
    distance = sub_two_vector(id_img, live_img, bbox_id, landmark_id, bbox_live, landmark_live)
    prob = model.predict([np.array(distance)])
    print(44444444)
    result = ""
    if(prob[0] == "1"):
        result = "Matching"
    else:
        result = "No Matching"
    #print("dddddddddddddddddddddddddddđsdsds: ",result)
    return result, id_crop, live_crop

print(predict_matching(args.id_image,args.live_image))
