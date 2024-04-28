from keras.models import load_model
import cv2
import numpy as np
from keras_preprocessing.image import img_to_array, load_img
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

model_character = load_model(r'D:\Code_school_nam3ki2\TestModel\model_license_plate_v11.h5')

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V','X', 'Y', 'Z']
le = LabelEncoder()
encode_label = le.fit_transform(labels)

# Chuyển đổi thành dạng nhị phân
binary_labels = to_categorical(encode_label)

dict = {}

for label in binary_labels:
    dict[labels[np.argmax(label)]] = label

test_folder_path = r'D:\Character_dataset\LPALIC_Latin\LPALIC_Latin\XTrain_All_Collection'
def getImageTrain(dirImg,lstImg,target_size=(40, 30)):        
        for filename in os.listdir(dirImg):
            filename_paths = os.path.join(dirImg,filename)   
            lst_filename_path = []
            for filename_path in os.listdir(filename_paths):   
                data_path = os.path.join(filename_paths,filename_path) 
                img = load_img(data_path, target_size=target_size, color_mode='grayscale')
                img_array = img_to_array(img)
                img_array /=255
                label = data_path.split('\\')[-2]           
                lst_filename_path.append((img_array,dict[label]))
            lstImg.extend(lst_filename_path) 
        return lstImg
 

data_test = getImageTrain(test_folder_path, [], target_size=(30, 40))
x_test = [x[0] for x in data_test]
y_test = [y[1] for y in data_test]

x_test = tf.convert_to_tensor(x_test)
val_loss, val_acc = model_character.evaluate(x_test, np.array(y_test))
print("Loss: ", val_loss)
print("Accuracy: ", val_acc * 100)