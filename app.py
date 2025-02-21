from flask import Flask,render_template,request,jsonify
import requests
import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model
import logging
from src.configuration import setup_logging


setup_logging()

# Initialize the Flask application
app = Flask(__name__)

# Loading pre-trained model
model = load_model('cnn_model.keras')
logging.info("trained model is loaded")


@app.route("/")
def home():
    
    """This route renders the homepage where users can input data to predict brain tumor."""
    return render_template('home.html')





@app.route("/predict",methods=['POST'])
def predict():
    
    """
    This route processes the input image submitted by the user, applies necessary transformations,
    runs predictions using trained model,and returns the prediction along with its percentage in the form of graph.
   """
    
    
    image_url = request.form["url"]
    
    image = request.files['image-upload']
    
    if image_url!='':
        try:
            response = requests.get(image_url)
            img = np.array(bytearray(response.content),np.uint8)
            logging.info("url found")
        except:
            return render_template('home.html',prediction_text="Please upload valid image url.")    
            
    elif image !='':
        img_bytes = image.read()
        img = np.frombuffer(img_bytes, np.uint8)
        logging.info("image uploaded")
    else:
        return render_template('home.html',prediction_text="file or link not uploaded.")    
             
    try: 
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img,(256,256))
        img= img.astype('float32')/255.0
        img = np.expand_dims(img,axis=0)
        pred = model.predict(img)
        logging.info(' input image preprocessed')
        print(pred)
        print(max(pred[0])*100)
        index = np.argmax(pred[0])
        print(index)
        labels = ['glioma','meningioma','not brain tumor','pituitary']
        
        logging.info("output message is returned")  
        
        if index==0:
           pred_text="The person is having brain tumor of type glioma."
           return jsonify({
              'labels':labels,
               'predictions': [float((max(pred[0])*100)),0,0,0],
                'predictionText':pred_text })
        elif index==1:
            pred_text= "The person is having brain tumor of type meningioma."
            return jsonify({
              'labels': labels,
               'predictions': [0,float((max(pred[0])*100)),0,0],
                'predictionText':pred_text })
        elif index==2:
             pred_text="The person is not having brain tumor"
             return jsonify({
              'labels': labels,
               'predictions': [0,0,float((max(pred[0])*100)),0],
                'predictionText':pred_text })
        else:
             pred_text="The person is having brain tumor of type pituitary"
             return jsonify({
              'labels': labels,
               'predictions': [0,0,0,float((max(pred[0])*100))],
                'predictionText':pred_text })


        
    except Exception  as e:
        print(e)
        return  render_template('home.html',prediction_text='Error in preprocessing the image')
    

    
# Run the Flask app in debug mode for local testing  
if __name__ == '__main__':  
       app.run(debug=True)
