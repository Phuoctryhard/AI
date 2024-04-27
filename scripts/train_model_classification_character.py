import os
import numpy as np
import torch
from keras import  Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt

IMG_DATA = 'CNN letter Dataset/CNN letter Dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)
root_path = os.getcwd()

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
le = LabelEncoder()
encode_label = le.fit_transform(labels)

# Chuyển đổi thành dạng nhị phân
binary_labels = to_categorical(encode_label)

dict = {}

for label in binary_labels:
    dict[labels[np.argmax(label)]] = label

def getImageTrain(dirImg,lstImg,target_size=(30, 40)):        
        for i, filename in enumerate(os.listdir(dirImg), 1):
            percentage = (i / len(os.listdir(dirImg))) * 100
            print(f"{filename} - {percentage:.2f}%")
            filename_paths = os.path.join(dirImg,filename)   
            lst_filename_path = []
            for filename_path in os.listdir(filename_paths):   
                data_path = os.path.join(filename_paths,filename_path) 
                img = load_img(data_path, target_size=target_size, color_mode='grayscale')
                img_array = img_to_array(img)
                img_array /= 255
                label = data_path.split('\\')[-2]           
                lst_filename_path.append((img_array,dict[label]))
            lstImg.extend(lst_filename_path) 
        return lstImg 

dir_folder = os.path.join(root_path, IMG_DATA)
subfolders = [folder for folder in os.listdir(dir_folder) if os.path.isdir(os.path.join(dir_folder, folder))]
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

X_data = getImageTrain(IMG_DATA, [])
np.random.shuffle(X_data)

X = np.array([x[0] for _, x in enumerate(X_data)])
Y = np.array([y[1] for _, y in enumerate(X_data)])

x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=True, test_size=.2, random_state=42)
X_train, x_val, Y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=.2, random_state=42)

print("Training dataset shape: ", X_train.shape, Y_train.shape)
print("Validation dataset shape: ", x_val.shape, y_val.shape)
print("Testing dataset shape: ", x_test.shape, y_test.shape)

model = Sequential()
model.add(Conv2D(16, (3,3), input_shape=(30, 40, 1), activation='relu', padding='same'))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(35, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, steps_per_epoch=len(X_train) // 32, epochs=20, validation_data=(x_val, y_val), validation_steps=len(x_val) // 32, callbacks=[early_stopping])

model.save('model_license_plate_v11.h5') 
history = hist.history
plt.figure(figsize=(8,6))
plt.plot(history['accuracy'], color='blue', label='train_accuracy')
plt.plot(history['val_accuracy'], color='red', label='val_accuracy')
plt.legend(loc='lower right')
plt.show()

val_loss, val_acc = model.evaluate(x_test, y_test)
print(f"model loss: {val_loss}, model accuracy: {val_acc}")












