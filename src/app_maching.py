from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
# from pymongo import MongoClient
import os
import imutils
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import cv2
import json
from keras.models import load_model
from flask import Flask, request, json, redirect, url_for
from werkzeug.utils import secure_filename
from flask import render_template
from predict_matching import predict_matching
from PIL import Image

app = Flask(__name__, static_url_path='/static')

@app.route("/", methods=['POST', 'GET'])
def index():
    if(request.method == 'GET'):
        return render_template('matching.html', result = "", id_crop= '')
    else:
        if not os.path.exists("static/file_client/"):
            os.makedirs("static/file_client/")
        id_img = request.files['id_img']
        id_image = Image.open(id_img)
        id_non_image = secure_filename(id_img.filename)
        id_image.save('static/file_client/'+id_non_image)
        live_img = request.files['live_img']
        live_image = Image.open(live_img)
        live_non_image = secure_filename(live_img.filename)
        live_image.save('static/file_client/'+live_non_image)
        result, id_crop, live_crop = predict_matching("static/file_client/" + str(id_non_image),"static/file_client/" + str(live_non_image))
        return render_template('matching.html', result=result, id_crop=id_crop, live_crop=live_crop)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)

