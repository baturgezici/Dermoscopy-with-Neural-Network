# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:45:10 2021

@author: Batur
"""
from tkinter import *
from PIL import ImageTk, Image, ImageOps
from tkinter import filedialog
from numpy import asarray
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import model_from_json
import numpy as np

label_mapping={
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'
}

json_file = open("model.json","r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

loaded_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

root = Tk()
root.geometry("550x300")

def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def open_img():
    x = openfn()
    img = Image.open(x)    
    imga = cv2.imread(x,0)
    resized = cv2.resize(imga, (28,28),interpolation=cv2.INTER_AREA)
    numpyarray = (asarray(resized)/255).reshape(-1,28,28,1)
    prediction = loaded_model.predict(numpyarray)
    disease = label_mapping[np.argmax(prediction)]
    disease_accuracy = str(prediction[0][np.argmax(prediction)])
    T = Text(root,height=5,width=30)
    T.place(x=265,y=15)
    T.insert(END, disease + " = " + disease_accuracy)
    # index = numpy.where(loaded_model.predict(numpyarray) == numpy.amax(loaded_model.predict(numpyarray)))
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.place(anchor=NW,x=5,y=5)

btn = Button(root, text='chose image', command=open_img).place(x=450,y=250)

root.resizable(width=0,height=0)
root.mainloop()