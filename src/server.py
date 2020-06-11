from flask import Flask, jsonify, request
import base64
import io
from matching_face import predict_matching, extract_feature, norm_vec, face_embedding
import numpy as np
import cv2
from PIL import Image

app = Flask(__name__)

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

# route http posts to this method
@app.route('/face/verification', methods=['POST'])
def test():
    try:
        body_json = request.get_json()
        
        id_img = toRGB(stringToImage(body_json['image_cmt']))
        selfie_img = toRGB(stringToImage(body_json['image_live']))
        
        res, id_bbox, selfie_bbox, embedding_id, embedding_selfie, time_taken = predict_matching(id_img, selfie_img)
        res_string = ""
        res_int = -1
        if(res[1] > 0.75):
            res_string = "same person"
            res_int = 2
        elif(res[1] < 0.25):
            res_string = "not same"
            res_int = 0
        else:
            res_string = "may be same"
            res_int = 1
        mem =  {"api_version": "1.0.4" ,"error_message": content, "error_code": code, "copy_right": "Copyright©2018-2019 "}
        return jsonify(
            verify_result = res_int,
            verify_result_text = res_string,
            sim = res[1],
            message = mem,
            verification_time = time_taken,
            face_loc_cmt = id_bbox,
            face_loc_live = selfie_bbox,
            feature_vector_face_cmt = embedding_id,
            feature_vector_face_live = embedding_selfie
        )
    except Exception as err:
        print("Error occurred")
        print(err)
        return("Error, image not received.")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)