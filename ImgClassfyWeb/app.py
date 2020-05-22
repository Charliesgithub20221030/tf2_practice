import flask 
import werkzeug 
import os 
import execute 
import getConfig
import requests 
import pickle
from flask import request, jsonify
import numpy as np
from  PIL import Image 

gConfig = {}
gConfig = getConfig.get_config(config_file= 'config.init')

app = flask.Flask('imgClassifierWeb')

def CNN_predict():
    file = gConfig['dataset_path'] + 'batchs.meta'
    patch_bin_file = open(file ,"rb")
    label_name_dict = pickle.load(patch_bin_file)['label_names']

    global secure_filename 
    img = Image.open(os.path.join(app.root_path,secure_filename))

    r,g,b = img.split()

    img=np.concatenate((
        np.array(r),
        np.array(g),
        np.array(b)
        ))

    image = img.reshape([1,32,32,3])/255

    predicted_class = execute.predict(image)

    return flask.render_template(
            template_name_or_list='prediction_result.html',
            predicted_class = predicted_class)

def upload_image():
    global secure_filename
    if flask.request.method =='POST':
        img_file = flask.request.files['image_file']
        secure_filename = werkzeug.secure_filename(img_file.filename)
        img_path = os.path.join(app.root_path,secure_filename)
        img_file.save(img_path)
        print("Successfully upload image")

        return flask.redirect(flask.url_for(endpoint = "predict"))
    print("Failed to upload image")

def redirect_upload():
    return flask.render_template(template_name_or_list='upload_image.html')
            



app.add_url_rule(rule=  '/predict/',
        endpoint = 'predict',
        view_func= CNN_predict)

app.add_url_rule(rule = '/upload/',
        endpoint ="upload",
        view_func = upload_image, methods =['POST'])

app.add_url_rule(rule = '/',
        endpoint = 'homepage',
        view_func = redirect_upload)


if __name__=='__main__':
    app.run(host = '0.0.0.0', port = 7777 , debug = False )


