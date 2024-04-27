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

def delete_img_dir(dir_path, label):
    for img in os.listdir(os.path.join(dir_path, label)):
        if img.startswith('generate'):
            img_path = os.path.join(dir_path, label, img)
            os.remove(img_path)

clearDir(r'D:\Code_school_nam3ki2\TestModel\letter_dataset_result')
# list_label_need = ['4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

# list_img_need = [
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\4\aug22424_3.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\5\aug23443_2.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\6\aug24084_3.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\7\aug25355_4.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\8\aug25888_7.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\9\aug26745_4.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\A\aug80_19.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\B\aug989_8.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\C\aug505_4.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\D\aug724_3.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\E\aug1074_13.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\F\aug1320_19.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\G\aug4704_3.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\H\aug2519_18.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\K\aug3289_8.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\L\aug3567_6.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\M\aug3722_1.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\N\aug9438_7.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\P\aug10754_3.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\S\aug12239_8.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\T\aug5885_4.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\U\aug13603_2.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\V\aug15187_6.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\X\aug6486_5.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\Y\aug7210_9.jpg",
#     r"D:\Code_school_nam3ki2\TestModel\CNN letter Dataset\CNN letter Dataset\Z\aug7504_3.jpg"
# ]

# list_label_not_flip = ['4', '5', '6', '7', '9', 'B', 'C', 'D', 'E', 'F', 'G', 'K', 'L', 'N', 'P', 'S', 'U', 'Z']
# list_label_can_flip = ['8', 'A', 'H', 'M', 'T', 'V', 'X', 'Y']
    
# for i in range(len(list_label_need)):
#     if list_label_need[i] in list_label_not_flip:
#         datagen.horizontal_flip = False
#         augment_images(dir_path, datagen, list_label_need[i], list_img_need[i])
#     else:
#         datagen.horizontal_flip = True
#         augment_images(dir_path, datagen, list_label_need[i], list_img_need[i])
        