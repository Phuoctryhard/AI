import os
import numpy as np
import torch
from keras import  Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from keras_preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
IMG_DATA = 'CNN letter Dataset/CNN letter Dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)


labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
le = LabelEncoder()
encode_label = le.fit_transform(labels)

# Chuyển đổi thành dạng nhị phân
binary_labels = to_categorical(encode_label)

dict = {}

for label in binary_labels:
    dict[labels[np.argmax(label)]] = label

def getImageTrain(dirImg,lstImg,target_size=(30, 40)):        
        for filename in os.listdir(dirImg):
            filename_paths = os.path.join(dirImg,filename)   
            lst_filename_path = []
            for i, filename_path in enumerate(os.listdir(filename_paths), 1):   
                percentage = i / len(os.listdir(filename_paths)) * 100
                print(f' label {filename}, {percentage:.2f}%')
                data_path = os.path.join(filename_paths,filename_path) 
                img = load_img(data_path, target_size=target_size, color_mode='grayscale')
                img_array = img_to_array(img)
                img_array /=255
                label = data_path.split('\\')[-2]           
                lst_filename_path.append((img_array,dict[label]))
            lstImg.extend(lst_filename_path) 
        return lstImg 

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

X_data = getImageTrain( IMG_DATA, [])
np.random.shuffle(X_data)


X = np.array([x[0] for _, x in enumerate(X_data)])
Y = np.array([y[1] for _, y in enumerate(X_data)])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, x_valid, Y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

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
model.add(Dense(30, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, steps_per_epoch=len(X_train) // 32, epochs=20, validation_data=(x_valid, y_valid), validation_steps=len(x_valid) // 32, callbacks=[early_stopping])
 
model.save('model_license_plate_v11.h5') 

val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss: ", val_loss)
print("Accuracy: ", val_acc)











