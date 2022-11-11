import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np
import cv2


model_dir = 'static/imageclassifier_2.h5'

#Function to show Image
def load_image(image_file):
    img = Image.open(image_file)
    img_array = np.array(img)
    cv2.imwrite('out.jpg', img_array)

    return img

def pass_model():
    #Load the model
    model = tf.keras.models.load_model(model_dir)
    #Read image
    img = cv2.imread('out.jpg')
    resize = tf.image.resize(img, (256,256))
    #Predict
    yhat = model.predict(np.expand_dims(resize/255, 0))
    return yhat

with st.sidebar:
    # st.image()
    st.title("X-Detect")
    st.info("Detecting X")

image_file = st.file_uploader("Upload your Image", type=["png","jpg","jpeg","BMP"])

if image_file is not None:
    
    #To See details
    file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
    st.write(file_details)

    # To View Uploaded Image
    st.image(load_image(image_file),width=250)
    

    #Run Prediction
    predict= pass_model()
    if predict > 0.5: 
        st.write(f'Predicted class is Good')
    else:
        st.write(f'Predicted class is NG')
