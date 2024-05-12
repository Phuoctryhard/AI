import os
import re
import os
import glob
import cv2
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt

dir_path = r'D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset'
subfolders = [subfolder for subfolder in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subfolder))]
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

# Define the augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    data_format='channels_last',
    brightness_range=[0.5, 1.5]
)

def clearDir(dir_path):
    subfolder = [subfolder for subfolder in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, subfolder))]
    for sub in subfolder:
        sub_path = os.path.join(dir_path, sub)
        for file in os.listdir(sub_path):
            os.remove(os.path.join(sub_path, file))
            
def remove_folder_not_in_label(folder):
    subfolder = [subfolder for subfolder in os.listdir(folder) if os.path.isdir(os.path.join(folder, subfolder))]
    for sub in subfolder:
        if sub not in labels:
            sub_path = os.path.join(folder, sub)
            os.rmdir(sub_path)

def create_folder(folder):
    for label in labels:
        label_path = os.path.join(folder, label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
    
def Check_file_cnt_in_folder(folder):
    subfolder = [subfolder for subfolder in os.listdir(folder) if os.path.isdir(os.path.join(folder, subfolder))]
    for sub in subfolder:
        sub_path = os.path.join(folder, sub)
        print(f"{sub}: {len(os.listdir(sub_path))}")    

def augment_images(dir_path, datagen, label, image_path):
    label_path = os.path.join(dir_path, label)
    total_cnt = 2000
    img_path = os.path.join(label_path, image_path)
    img = load_img(img_path)
    x = img_to_array(img)
    x = cv2.resize(x, (75, 100))
    x = x.reshape((1,) + x.shape)
    for batch in datagen.flow(x, batch_size=1, save_to_dir=label_path, save_prefix='generate', save_format='jpg'):
        if len(os.listdir(label_path)) == total_cnt:
            break
        
        
def augment_dir_images(dir_path, datagen, labels):
    total_generate_per_img = 5
    total_img = 2000
    for pc, label in enumerate(labels, 1):
        label_path = os.path.join(dir_path, label)
        for idx, img_name in enumerate(os.listdir(os.path.join(dir_path, label)), 1):
            if len(os.listdir(label_path)) == total_img:
                break
            percent = idx / total_img *100
            print(f"Processing: {label}, {percent:.2f}%")
            i = 0
            img_path = os.path.join(dir_path, label, img_name)
            img_arr = load_img(img_path)
            img_arr = img_to_array(img_arr)
            img_arr = cv2.resize(img_arr, (75, 100))
            img_arr = img_arr.reshape((1,) + img_arr.shape)
            for batch in datagen.flow(img_arr, batch_size=1, save_to_dir=os.path.join(dir_path, label), save_prefix='generate', save_format='jpg'):
                i += 1
                if i == total_generate_per_img or len(os.listdir(label_path)) == total_img:
                    break
            

def delete_img_dir(dir_path, label):
    for img in os.listdir(os.path.join(dir_path, label)):
        if img.startswith('generate'):
            img_path = os.path.join(dir_path, label, img)
            os.remove(img_path)

# list_label_not_flip = ['1', '2', '3', '4', '5', '6', '7', '9', 'B', 'C', 'D', 'E', 'F', 'G', 'K', 'L', 'N', 'P', 'S', 'U', 'Z']
# list_label_can_flip = ['0','8', 'A', 'H', 'M', 'T', 'V', 'X', 'Y']

path = r'D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset'
Check_file_cnt_in_folder(path)