from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')


def read_image(filename):
	# Load the image
	img = load_img(filename, target_size=(224,224))
	# Convert the image to array
	img = img_to_array(img)
	# Reshape the image into a sample of 3 channel with 1 image
	img = img.reshape(1, 224, 224, 3)
	# Prepare it as pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        file=request.files['file']
        try:
        # Save the file to ./uploads
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('uploads', filename)
                file.save(file_path)
                img = read_image(file_path)
                #loading the model and predicting  
                model11=load_model('VGG16.h5')
                preds=model11.predict(img)
                labels_dict={0:'Goggles',1:'without_mask',2:'with_mask'}
                label=np.argmax(preds,axis=1)[0]
                result=labels_dict[label]
                return result
        except:
            return "Unable to read the file. Please check if the file extension is correct."   
            
    return None


if __name__ == '__main__':
    
    app.run()
