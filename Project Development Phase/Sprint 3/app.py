import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

app = Flask(__name__)
model=load_model("model/number.h5")

@app.route('/')
def home() :
    return render_template('home.html')

@app.route('/getdata',methods=['GET','POST'])
def upload() :
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__name__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['.', '*', '8', '=', '5', '4','-', '9', '1', '+', '7', '6', '3', '*', '2', '0']
        text="Your Output is : "+str(index[pred[0]])
    return render_template('output.html',output=text)

if __name__=='__main__' :
        app.run()
