import requests
import json
import cv2



label_dict = {0.0:"Tomato___Bacterial_spot",1.0:"Tomato___Early_blight",2.0:"Tomato___Late_blight",3.0:"Tomato___Leaf_Mold",
              4.0:"Tomato___Septoria_leaf_spot", 5.0:"Tomato___Spider_mites Two-spotted_spider_mite", 6.0:"Tomato___Target_Spot",
              7.0:"Tomato___Tomato_Yellow_Leaf_Curl_Virus", 8.0:"Tomato___Tomato_mosaic_virus",9.0:"Tomato___healthy"
              }

def _prediction_my_model(data):
    headers = {"content-type": "application/json"}
    payload = {"instances":[{"raw_image":data.tolist()}]}
    json_response = requests.post(url='http://localhost:8501/v1/models/tomato_model:predict',json=payload ,headers=headers)
    json_result = json.loads(json_response.text)["predictions"]
    return json_result

img = cv2.imread('imagefile.jpg')    
img = cv2.resize(img,(100,100))
pred = _prediction_my_model(img)
print("Prediction output is : ",label_dict[pred[0][0]])