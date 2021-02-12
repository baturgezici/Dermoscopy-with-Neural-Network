# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:51:29 2021

@author: Batur
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv(r"D:\DERMANET\HAM10000\hmnist_28_28_L.csv")

#Preprocess

y = df["label"].copy()
x = df.drop("label", axis=1).copy()

label_mapping={
    0: 'nv',
    1: 'mel',
    2: 'bkl',
    3: 'bcc',
    4: 'akiec',
    5: 'vasc',
    6: 'df'
}

#Rescale-normalize for input process
alt=x
x = np.array(x/255).reshape(-1,28,28,1)

#Showing diseases with pictures

sample_data = pd.Series(list(zip(x, y))).sample(9)

sample_x = np.stack(np.array(sample_data.apply(lambda x: x[0])))
sample_y = np.array(sample_data.apply(lambda x: x[1]))

plt.figure(figsize=(23, 23))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(np.squeeze(sample_x[i]))
    img_label = label_mapping[sample_y[i]]
    plt.title(img_label)
    plt.axis("off")

plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=46)

# my CNN architecture: In -> [Conv2D]*4 -> AvgPool -> Output

inputs = tf.keras.Input(shape=(28, 28, 1), name='input')

conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, name='conv1', padding='same')(inputs)
maxpool1 = tf.keras.layers.MaxPooling2D(name='maxpool1')(conv1)

conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, name='conv2', padding='same')(maxpool1)
maxpool2 = tf.keras.layers.MaxPooling2D(name='maxpool2')(conv2)

conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, name='conv3', padding='same')(maxpool2)
maxpool3 = tf.keras.layers.MaxPooling2D(name='maxpool3')(conv3)

conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, name='conv4', padding='same')(maxpool3)
maxpool4 = tf.keras.layers.MaxPooling2D(name='maxpool4')(conv4)

avgpool = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(maxpool4)

outputs = tf.keras.layers.Dense(7, activation='softmax', name='output')(avgpool)


model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())
tf.keras.utils.plot_model(model)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


batch_size = 32
epochs = 50

history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

model_acc = model.evaluate(x_test, y_test, verbose=0)[1]
model_acct = model.evaluate(x_test, y_test, verbose=0)

print("Test Accuracy is: {:.3f}%".format(model_acc * 100))

df2 = pd.read_csv(r"D:\DERMANET\HAM10000\HAM10000_metadata.csv")
g = sns.catplot(x="dx", kind="count", hue="dx_type", palette="bright", data=df2)
g.fig.set_size_inches(16, 5)

g.ax.set_title('Skin Cancer by Histopathology', fontsize=20)
g.set_xlabels('Type of Skin Cancer', fontsize=14)
g.set_ylabels('Number of Occurence', fontsize=14)
g._legend.set_title('Histopathology Type')
g = sns.catplot(x="dx", kind="count", hue="localization", palette='bright', data=df2)
g.fig.set_size_inches(16, 9)

g.ax.set_title('Skin Cancer Localization', fontsize=20)
g.set_xlabels('Type of the cancer', fontsize=14)
g.set_ylabels('Number of occurence', fontsize=14)
g._legend.set_title('Localization')
y_true = np.array(y_test)

y_pred = model.predict(x_test)
y_pred = np.array(list(map(lambda x: np.argmax(x), y_pred)))

cm = confusion_matrix(y_true, y_pred)
clr = classification_report(y_true, y_pred, target_names=label_mapping.values())

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False, cmap='Blues')

plt.xticks(np.arange(7) + 0.5, label_mapping.values())
plt.xlabel("Predicted")

plt.yticks(np.arange(7) + 0.5, label_mapping.values())
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()
print(clr)

#%%
# SAVING THE MODEL AND WEIGHTS
import h5py
from keras.models import model_from_json

model_json = model.to_json()

with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")

json_file.close()
#%%
# LOADING MODEL AND WEIGHTS
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import model_from_json
import numpy

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


#%%

from tkinter import *
from PIL import ImageTk, Image, ImageOps
from tkinter import filedialog
from numpy import asarray
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

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