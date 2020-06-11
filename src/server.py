from flask import Flask, jsonify, request
import base64
import io
from matching_face import predict_matching, extract_feature, norm_vec, face_embedding
import numpy as np
import cv2

app = Flask(__name__)

# route http posts to this method
@app.route('/face/verification', methods=['POST'])
def test():
    try:
        # check if the post request has the file part
        file_dict = request.files.to_dict()
        print(file_dict.keys())
        id_img = file_dict['id_image'].read()
        selfie_img = file_dict['selfie_image'].read()
        #convert string data to numpy array
        npimg_id = np.fromstring(id_img, np.uint8)
        npimg_selfie = np.fromstring(selfie_img, np.uint8)
        # convert numpy array to image
        id_img = cv2.imdecode(npimg_id, cv2.IMREAD_COLOR)
        selfie_img = cv2.imdecode(npimg_selfie, cv2.IMREAD_COLOR)
        
        res = predict_matching(id_img, selfie_img)
        res_string = ""
        if(res == 1):
            res_string = "Matching"
        elif(res == -1):
            res_string = "No face found"
        else:
            res_string = "No matching"
        return jsonify(
            result = res_string
        )
    except Exception as err:
        print("Error occurred")
        print(err)
        return("Error, image not received.")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)