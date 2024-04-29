import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from keras import models
import numpy as np
import re

config_path = 'config\\pipeline.config'
model_character_path = r'D:\Code_school_nam3ki2\TestModel\model\classfication_character\model_license_plate_v9.h5'
label_path = 'config\\label_map.pbtxt'
checkpoint_path = 'model\\detect_liscense_plate'
test_folder_path = r'D:\car_long'
result_folder_path = r'D:\Code_school_nam3ki2\TestModel\images_result'
improve_img_resolution_file_path = r'D:\Code_school_nam3ki2\TestModel\model\improve_image_resolution\test.py'

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(root_path, label_path))
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(os.path.join(root_path, config_path))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(root_path, checkpoint_path, 'ckpt-11')).expect_partial()

model_character = models.load_model(model_character_path)

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def pre_process(img, W):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(img_gray, 11, 17, 17) #Noise reduction
    # blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, int(W/10) if int(W/10)%2 !=0  else int(W/10)+1, 10)
    binary = cv2.adaptiveThreshold(bfilter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, int(W/20) if int(W/20)%2 !=0  else int(W/20)+1, 15)
    return binary

def predict_license_plate(img):
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    image_np_crop = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'], 
                detections['detection_classes'] + label_id_offset, #Cộng vô để khớp với category_index
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True, #Chuấn hóa về 0 => 1
                max_boxes_to_draw=1, 
                agnostic_mode=False) #Tất cả các hộp đều được vẽ cùng màu
    return image_np_with_detections, image_np_crop, detections['detection_boxes'][0]

def get_detections(img, detections):
    img_crop = None
    ymin, xmin, ymax, xmax = detections
    height, width, _ = img.shape
    ymin = int(ymin * height)
    ymax = int(ymax * height)
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    X = xmin
    Y = ymin
    W = xmax - xmin
    H = ymax - ymin
    if X != None and Y != None and W != None and H != None:
        #Cắt vùng chứa biển số xe
        img_crop = img[int(Y)-10: int(Y)+int(H) + 10, int(X)-10: int(X)+int(W)+10] if int(Y)-5 > 0 and int(X)-5 > 0 else img[int(Y): int(Y)+int(H), int(X): int(X)+int(W)]
        cv2.imwrite(os.path.join(result_folder_path, "cropped_image.jpg"), img_crop)
        img_crop = cv2.imread(os.path.join(result_folder_path, "cropped_image.jpg"))
        #Loại bỏ nhiễu 
        img_crop = cv2.fastNlMeansDenoisingColored(img_crop, None, 10, 10, 7, 21)
        
    return img_crop, X,Y,W,H

def segmentation_character(binary, img, img_crop, model_character, X, Y, W, H, type = 0):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print("Num_labels:", num_labels)
    candidates = []
    bounding_rects = []
    list_character = []
    i = 0
    for label in range(1, num_labels):
        # Tạo mask chứa các pixel có nhãn cùng là label
        mask = np.zeros(binary.shape, dtype=np.uint8)
        mask[labels == label] = 255 # Các các pixel cùng nhãn giá trị 255
        # Tìm contours từ mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc contours theo tiêu chí aspect ratio, solidity và height ratio
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)
            height_ratio = h / float(binary.shape[0])


            #Loại bỏ nhiễu dựa vào aspect ratio, solidity và height ratio
            if 0.2 < aspect_ratio < 0.8 and solidity > 0.1 and 0.2 < height_ratio < 1.0:
                bounding_rects.append((x, y, w, h))
                # Trích xuất ký tự
                character = img_crop[y-3: y + h+3, x-int(h*3/10-w/2):x + w+int(h*3/10-w/2)]
                candidates.append((x, y, character))

    if candidates[-1][1]/float(candidates[0][1]) > 2:  
        list_character_first = []
        list_character_second = []          
        first_lines = [item for item in candidates if (candidates[-1][1]/float(item[1])) >= 2.0 ]
        second_lines = [item for item in candidates if (candidates[-1][1]/float(item[1])) < 2.0 ]
        first_lines.sort(key=lambda item: item[0])
        first_lines = [item for item in first_lines]
        second_lines.sort(key=lambda item: item[0])
        second_lines = [item for item in second_lines]
        for first_line in first_lines:
                i+=1
                filename = f"region_{i}.jpg"
                save_path = os.path.join(result_folder_path, filename)
                cv2.imwrite(save_path, first_line[2])
                list_character_first.append(predict_image(save_path,model_character))

        for idx, (x, y,char)  in enumerate(first_lines):
        # Vẽ hình chữ nhật bao quanh ký tự
            cv2.rectangle(img, (x+int(X)-4, y+int(Y)-3), (x+int(X) + char.shape[1] - 6, y+int(Y) + char.shape[0] - 5), (0, 255, 0), 2)
        
        # Hiển thị ký tự lên ảnh
            cv2.putText(img, list_character_first[idx], (x+int(X)-3, y + int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   
                    
        for second_line in second_lines:
                i+=1
                filename = f"region_{i}.jpg"
                save_path = os.path.join(result_folder_path, filename)
                cv2.imwrite(save_path, second_line[2])
                list_character_second.append(predict_image(save_path,model_character))
        list_character.extend(list_character_first)
        list_character.extend(list_character_second)
        print("".join(list_character))

        for idx, (x, y,char)  in enumerate(second_lines):
        # Vẽ hình chữ nhật bao quanh ký tự
            cv2.rectangle(img, (x+int(X)-4, y+int(Y)-3), (x+int(X) + char.shape[1] - 6, y+int(Y) + char.shape[0] - 5), (0, 255, 0), 2)
        
        # Hiển thị ký tự lên ảnh
            cv2.putText(img, list_character_second[idx], (x+int(X)-3, y + int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị ảnh với các ký tự đã nhận dạng được vẽ lên
        cv2.imshow("Detected Characters", img)
    else:
        candidates.sort(key=lambda item: item[0])
        candidates = [item for item in candidates]     
        for character in candidates:
                i+=1
                filename = f"region_{i}.jpg"
                save_path = os.path.join(result_folder_path, filename)
                cv2.imwrite(save_path, character[2])
                list_character.append(predict_image(save_path,model_character))

        print("".join(list_character))
        for idx, (x, y,char)  in enumerate(candidates):
        # Vẽ hình chữ nhật bao quanh ký tự
            cv2.rectangle(img, (x+int(X)-4, y+int(Y)-3), (x+int(X) + char.shape[1] - 6, y+int(Y) + char.shape[0] - 5), (0, 255, 0), 2)
    
        # Hiển thị ký tự lên ảnh
            cv2.putText(img, list_character[idx], (x+int(X)-3, y + int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Hiển thị ảnh với các ký tự đã nhận dạng được vẽ lên
        cv2.imshow("Detected Characters", img)
    if type == 0:
        cv2.waitKey()
        cv2.destroyAllWindows()

def segmentation_character_test(binary, img, img_crop, model_character, X, Y, W, H, type = 0):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print("Num_labels:", num_labels)
    candidates = []
    bounding_rects = []
    list_character = []
    i = 0
    for label in range(1, num_labels):
        # Tạo mask chứa các pixel có nhãn cùng là label
        mask = np.zeros(binary.shape, dtype=np.uint8)
        mask[labels == label] = 255 # Các các pixel cùng nhãn giá trị 255
        # Tìm contours từ mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lọc contours theo tiêu chí aspect ratio, solidity và height ratio
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            solidity = cv2.contourArea(contour) / float(w * h)
            height_ratio = h / float(binary.shape[0])


            #Loại bỏ nhiễu dựa vào aspect ratio, solidity và height ratio
            if 0.2 < aspect_ratio < 0.8 and solidity > 0.1 and 0.2 < height_ratio < 1.0:
                bounding_rects.append((x, y, w, h))
                # Trích xuất ký tự
                character = img_crop[y-3: y + h+3, x-int(h*3/10-w/2):x + w+int(h*3/10-w/2)]
                candidates.append((x, y, character))

    if candidates[-1][1]/float(candidates[0][1]) > 2:  
        list_character_first = []
        list_character_second = []          
        first_lines = [item for item in candidates if (candidates[-1][1]/float(item[1])) >= 2.0 ]
        second_lines = [item for item in candidates if (candidates[-1][1]/float(item[1])) < 2.0 ]
        first_lines.sort(key=lambda item: item[0])
        first_lines = [item for item in first_lines]
        second_lines.sort(key=lambda item: item[0])
        second_lines = [item for item in second_lines]
        for first_line in first_lines:
                i+=1
                filename = f"region_{i}.jpg"
                save_path = os.path.join(result_folder_path, filename)
                cv2.imwrite(save_path, first_line[2])
                list_character_first.append(predict_image(save_path,model_character))

        for idx, (x, y,char)  in enumerate(first_lines):
        # Vẽ hình chữ nhật bao quanh ký tự
            cv2.rectangle(img, (x+int(X)-4, y+int(Y)-3), (x+int(X) + char.shape[1] - 6, y+int(Y) + char.shape[0] - 5), (0, 255, 0), 2)
        
        # Hiển thị ký tự lên ảnh
            cv2.putText(img, list_character_first[idx], (x+int(X)-3, y + int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   
                    
        for second_line in second_lines:
                i+=1
                filename = f"region_{i}.jpg"
                save_path = os.path.join(result_folder_path, filename)
                cv2.imwrite(save_path, second_line[2])
                list_character_second.append(predict_image(save_path,model_character))
        list_character.extend(list_character_first)
        list_character.extend(list_character_second)
        print("".join(list_character))

        for idx, (x, y,char)  in enumerate(second_lines):
        # Vẽ hình chữ nhật bao quanh ký tự
            cv2.rectangle(img, (x+int(X)-4, y+int(Y)-3), (x+int(X) + char.shape[1] - 6, y+int(Y) + char.shape[0] - 5), (0, 255, 0), 2)
        
        # Hiển thị ký tự lên ảnh
            cv2.putText(img, list_character_second[idx], (x+int(X)-3, y + int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Hiển thị ảnh với các ký tự đã nhận dạng được vẽ lên
        cv2.imshow("Detected Characters", img)
    else:
        candidates.sort(key=lambda item: item[0])
        candidates = [item for item in candidates]     
        for character in candidates:
                i+=1
                filename = f"region_{i}.jpg"
                save_path = os.path.join(result_folder_path, filename)
                cv2.imwrite(save_path, character[2])
                list_character.append(predict_image(save_path,model_character))

        print("".join(list_character))
        for idx, (x, y,char)  in enumerate(candidates):
        # Vẽ hình chữ nhật bao quanh ký tự
            cv2.rectangle(img, (x+int(X)-4, y+int(Y)-3), (x+int(X) + char.shape[1] - 6, y+int(Y) + char.shape[0] - 5), (0, 255, 0), 2)
    
        # Hiển thị ký tự lên ảnh
            cv2.putText(img, list_character[idx], (x+int(X)-3, y + int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Hiển thị ảnh với các ký tự đã nhận dạng được vẽ lên
        cv2.imshow("Detected Characters", img)
    if type == 0:
        cv2.waitKey()
        cv2.destroyAllWindows()

def predict_image(image_path, model):
    # img = improve_image_resolution(cv2.imread(image_path), "EDSR_x4.pb")
    img = cv2.imread(image_path)  
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (40,30))
    img_array = np.array(resized_img) 
    img_array = img_array/255.0
    img_input = img_array.reshape((-1, 30, 40, 1))
    pred = model.predict(img_input)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    predicted_label = labels[np.argmax(pred)]
    return predicted_label

def predict_image_2(img, model):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (40,30))
    img_array = np.array(resized_img) 
    img_array = img_array/255.0
    img_input = img_array.reshape((-1, 30, 40, 1))
    pred = model.predict(img_input)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    predicted_label = labels[np.argmax(pred)]
    return predicted_label

def image_detect(IMAGE_PATH, model_character):
    image_original = cv2.imread(IMAGE_PATH)
    image_np_with_detections, img, results = predict_license_plate(image_original)
    cv2.imshow('image_np_with_detections', image_np_with_detections)
    cv2.waitKey()
    cv2.destroyAllWindows()
    img_crop, X,Y,W,H = get_detections(img, results)
    cv2.imshow('img_crop', img_crop)
    cv2.waitKey()
    cv2.destroyAllWindows()
    binary = pre_process(img_crop, W)
    cv2.imshow('binary', binary)
    cv2.waitKey()
    cv2.destroyAllWindows()
    segmentation_character(binary, img, img_crop, model_character, X, Y, W, H)

def camera_detect(model_character):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        try:
            img_crop = None
            ret, img = cap.read()
            if not ret:
                print("Không thể đọc từ camera.")
                break
            img = cv2.resize(img, (800, 600))
            image_np_with_detections, img, results = predict_license_plate(img)
            cv2.imshow('image_np_with_detections', image_np_with_detections)
            img_crop, X,Y,W,H = get_detections(img, results)
            if img_crop is not None:
                binary = pre_process(img_crop, W)
                segmentation_character(binary, img, img_crop, model_character, X, Y, W, H, 1)
            else:
                print("Không tìm thấy biển số.")
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")

listIMGTest = os.listdir(test_folder_path)
listIMGTest = [os.path.join(test_folder_path, img) for img in listIMGTest]

for img_path in listIMGTest:
    image_detect(img_path, model_character)
