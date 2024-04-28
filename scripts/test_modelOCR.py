import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from keras import models
import cv2
import numpy as np
import re
import easyocr

reader = easyocr.Reader(['en'])
config_path = 'config\\pipeline.config'
model_character_path = r'D:\Code_school_nam3ki2\TestModel\model\classfication_character\model_license_plate_v9.h5'
label_path = 'config\\label_map.pbtxt'
checkpoint_path = 'model\\detect_liscense_plate'
test_folder_path = r'D:\extract_character'
result_folder_path = r'D:\Code_school_nam3ki2\TestModel\images_result'
improve_img_resolution_file_path = r'D:\Code_school_nam3ki2\TestModel\model\improve_image_resolution\test.py'
letter_extract_result_folder_path = r'D:\Code_school_nam3ki2\TestModel\letter_extract_result'
letter_dataset_result_folder_path = r'D:\Code_school_nam3ki2\TestModel\letter_dataset_result'

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(root_path, label_path))
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(os.path.join(root_path, config_path))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(root_path, checkpoint_path, 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def pre_process(img, W):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, int(W/10) if int(W/10)%2 !=0  else int(W/10)+1, 10)
    return binary

def predict_license_plate(IMAGE_PATH):
    img = cv2.imread(IMAGE_PATH)
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

def maximizeContrast(imgGrayscale):
    #Làm cho độ tương phản lớn nhất
    if len(imgGrayscale.shape) == 3:
        height, width, _ = imgGrayscale.shape
    else:
        height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10) #nổi bật chi tiết sáng trong nền tối
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10) #Nổi bật chi tiết tối trong nền sáng
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    #Kết quả cuối là ảnh đã tăng độ tương phản
    return imgGrayscalePlusTopHatMinusBlackHat

def get_detections(img, detections):
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
    img_crop = img[int(Y)-10: int(Y)+int(H) + 10, int(X)-10: int(X)+int(W)+10] if int(Y)-5 > 0 and int(X)-5 > 0 else img[int(Y): int(Y)+int(H), int(X): int(X)+int(W)]
    height, width, _ = img_crop.shape
    cv2.imwrite(os.path.join(result_folder_path, "cropped_image.jpg"), img_crop)

    if height < 100 and width < 100:
        command_improve_img_resolution = f"python {improve_img_resolution_file_path} {os.path.join(result_folder_path, 'cropped_image.jpg')} {os.path.join(result_folder_path, 'cropped_image.jpg')}"
        os.system(command_improve_img_resolution)

    img_crop = cv2.imread(os.path.join(result_folder_path, "cropped_image.jpg"))
    return img_crop, W

def predict_image(image_path, model):
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

def segmentation(IMAGE_PATH, model_character):
    image_np_with_detections, img, results = predict_license_plate(IMAGE_PATH)
    cv2.imshow('image_np_with_detections', image_np_with_detections)
    cv2.waitKey()
    cv2.destroyAllWindows()
    img_crop, W = get_detections(img, results)
    cv2.imshow('img_crop', img_crop)
    cv2.waitKey()
    cv2.destroyAllWindows()
    binary = pre_process(img_crop, W)
    cv2.imshow('binary', binary)
    cv2.waitKey()
    cv2.destroyAllWindows()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
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

            # Kiểm tra các điều kiện để loại bỏ nhiễu như dâu "." hay "-"
            if 0.2 < aspect_ratio < 0.8 and solidity > 0.1 and 0.2 < height_ratio < 1.0:
                bounding_rects.append((x, y, w, h))
                # Trích xuất ký tự
                y_start = max(0, y-3)
                y_end = min(binary.shape[0], y + h + 3)
                x_start = max(0, x-int(h*13/40-w/2))
                x_end = min(binary.shape[1], x + w+int(h*13/40-w/2))
                # Trích xuất ký tự
                character = img_crop[y_start:y_end, x_start:x_end]
                # character = img_crop[y-3: y + h+3, x-int(h*13/40-w/2):x + w+int(h*13/40-w/2)]
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
                    cv2.imwrite(filename, first_line[2])
                    list_character_first.append(predict_image(filename,model_character))

            for second_line in second_lines:
                    i+=1
                    filename = f"region_{i}.jpg"
                    cv2.imwrite(filename, second_line[2])
                    list_character_second.append(predict_image(filename,model_character))
            print(list_character_first)
            print(list_character_second)
        else:
            candidates.sort(key=lambda item: item[0])
            candidates = [item for item in candidates]     
            for character in candidates:
                    i+=1
                    filename = f"region_{i}.jpg"
                    cv2.imwrite(filename, character[2])
                    list_character.append(predict_image(filename,model_character))
            print(list_character)
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # print("Num_labels:", num_labels)
        # candidates = []
        # candidates_line1 = []
        # candidates_line2 = []
        # bounding_rects = []

    # # Sắp xếp lại stats theo tọa độ y để đảm bảo thứ tự từ trên xuống dưới
    # sorted_stats = sorted([stats[i] for i in range(1, num_labels)], key=lambda s: s[cv2.CC_STAT_LEFT])

    # for stat in sorted_stats:
    #     # Lấy thông tin từng ký tự
    #     x = stat[cv2.CC_STAT_LEFT]
    #     y = stat[cv2.CC_STAT_TOP]
    #     w = stat[cv2.CC_STAT_WIDTH]
    #     h = stat[cv2.CC_STAT_HEIGHT]
    #     area = stat[cv2.CC_STAT_AREA]

    #     aspect_ratio = w / float(h)
    #     solidity = area / float(w * h)
    #     height_ratio = h / float(binary.shape[0])

    #     # Điều chỉnh các điều kiện lọc dựa trên mô tả hình ảnh
    #     if 0.2 < aspect_ratio < 0.8 and solidity > 0.1 and 0.2 < height_ratio < 1:
    #         bounding_rects.append((x, y, w, h))
    #         # Điều chỉnh tọa độ để trích xuất ký tự
    #         y_start = max(0, y-5)
    #         y_end = min(binary.shape[0], y + h + 5)
    #         x_start = max(0, x - 3)
    #         x_end = min(binary.shape[1], x + w + 3)
    #         # Trích xuất ký tự
    #         character = img_crop[y_start:y_end, x_start:x_end]
    #         if y < binary.shape[0] // 2:
    #             candidates_line1.append(character)
    #         else:
    #             candidates_line2.append(character)

    # if len(candidates_line1) > 0 and len(candidates_line2) > 0:
    #     candidates = candidates_line1 + candidates_line2
    # else:
    #     candidates = candidates_line1 if len(candidates_line1) > 0 else candidates_line2     

# Define a function to extract the number from the filename
def extract_number(filename):
    x = re.search(r'\d+', filename)
    x = x.group() if x else -1
    return int(x)

# listImgTest = os.listdir(test_folder_path)
# listImgTest = [os.path.join(test_folder_path, img) for img in listImgTest if not img.endswith('.xml')]
# listImgTest.sort(reverse=True)
        

def test_predict_image(path, model_character_path):
    model_character = models.load_model(model_character_path)
    subdirectories = [name for name in os.listdir(letter_dataset_result_folder_path) if os.path.isdir(os.path.join(letter_dataset_result_folder_path, name))]
    for i, filename in enumerate(os.listdir(path), 0):
        percentage = (i+1)/len(os.listdir(path))*100
        print(f"Progress predict image: {percentage:.2f}%")
        filename_path = os.path.join(path, filename)
        predicted_label = predict_image(filename_path, model_character)
        if predicted_label in subdirectories:
            cv2.imwrite(os.path.join(letter_dataset_result_folder_path, predicted_label, filename), cv2.imread(filename_path, cv2.IMREAD_COLOR))
model = models.load_model(model_character_path)

list_character_need_extract = [
    'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]        

def extract_letter_from_dir(listImgTest):
    char_arr = []
    for i, IMAGE_PATH in enumerate(listImgTest, 0):
        #Calculate the progress percentage
        percentage = (i+1)/len(listImgTest)*100
        print(f"Progress extract character: {percentage:.2f}%, current image: {IMAGE_PATH}")
        try:
            candidates = segmentation(IMAGE_PATH)
            for j, candidate in enumerate(candidates, 0):
                char_arr.append(candidate)
        except Exception as e:
            print(f"An error occurred: {e}")

    for i, char in enumerate(char_arr, 0):
        percentage = (i+1)/len(char_arr)*100
        print(f"Progress save character: {percentage:.2f}%")
        c = predict_image_2(char, model)
        if c in list_character_need_extract:
            cv2.imwrite(os.path.join(letter_extract_result_folder_path, f"char_{i}.jpg"), char)



# extract_letter_from_dir(listImgTest)
# test_predict_image(letter_extract_result_folder_path, model_character_path)

# for IMAGE_PATH in listImgTest:
IMAGE_PATH = r"D:\extract_character\xemay2363_jpg.rf.56dec3d199b44526f62fed5dac123c62.jpg"
model = models.load_model(model_character_path)
# for IMAGE_PATH in listImgTest:
segmentation(IMAGE_PATH, model)
    