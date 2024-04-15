import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import models

config_path = 'config\\pipeline.config'
model_character_path = 'model\\classfication_character\\model_license_plate_v5.h5'
label_path = 'config\\label_map.pbtxt'
checkpoint_path = 'model\\detect_liscense_plate'

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def resize_image(image, max_size):
    h, w = image.shape[:2]
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def predict_image(image_path, model):
    img = cv2.imread(image_path)  
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (30, 40))
    img_array = np.array(resized_img) 
    img_array = np.expand_dims(img_array, -1)  # Thêm một chiều vào cuối
    img_input = np.expand_dims(img_array, 0)
    pred = model.predict(img_input)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    predicted_label = labels[np.argmax(pred)]
    return predicted_label

#Detect Image
category_index = label_map_util.create_category_index_from_labelmap(os.path.join(root_path, label_path))
IMAGE_PATH = os.path.join(r"D:\Code_school_nam3ki2\TestModel\Tensorflow\workspace\images\test\Cars402.png")

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

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'], 
            detections['detection_classes'] + label_id_offset, #Cộng vô để khớp với category_index
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True, #Chuấn hóa về 0 => 1
            max_boxes_to_draw=5, #Tối đa vẽ 5 hình
            agnostic_mode=False) #Tất cả các hộp đều được vẽ cùng màu


height, width, _ = image_np_with_detections.shape
# Cắt ảnh dựa trên tọa độ hộp giới hạn
ymin, xmin, ymax, xmax = detections['detection_boxes'][0]
ymin = int(ymin * height)
ymax = int(ymax * height)
xmin = int(xmin * width)
xmax = int(xmax * width)
cropped_image = image_np_with_detections[ymin:ymax, xmin:xmax]
model_character = models.load_model(os.path.join(root_path, model_character_path))
cropped_image = resize_image(cropped_image, 1000)
cv2.imshow("Crop_img", cropped_image)
cv2.waitKey()
cv2.imwrite(os.path.join(root_path, "images_result", "cropped_image.jpg"), cropped_image)
img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", img_gray)
cv2.waitKey()
_, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("binary", binary)
cv2.waitKey()

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
print("Num_labels:", num_labels)
candidates = []
# bounding_rects = []
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
        if 0.2 < aspect_ratio < 0.8 and solidity > 0.1 and 0.2 < height_ratio < 2.0:
            # bounding_rects.append((x, y, w, h))
            # Trích xuất ký tự
            character = cropped_image[y-3: y + h+3, x-3:x + w+3]
            candidates.append((x, y, character))
            
            
if candidates[-1][1]/float(candidates[0][1]) > 2:  
    list_character_first = []
    list_character_second = []          
    first_lines = [item for item in candidates if (candidates[-1][1]/float(item[1])) >= 2.0 ]
    second_lines = [item for item in candidates if (candidates[-1][1]/float(item[1])) < 2.0 ]
    first_lines.sort(key=lambda item: item[0])
    first_lines = [item[2] for item in first_lines]
    second_lines.sort(key=lambda item: item[0])
    second_lines = [item[2] for item in second_lines]
    for first_line in first_lines:
            i+=1
            filename = f"region_{i}.jpg"
            cv2.imwrite(os.path.join(root_path, "images_result", filename), first_line)
            list_character_first.append(predict_image(os.path.join(root_path, "images_result", filename),model_character))
                
    for second_line in second_lines:
            i+=1
            filename = f"region_{i}.jpg"
            cv2.imwrite(os.path.join(root_path, "images_result", filename), first_line)
            list_character_second.append(predict_image(os.path.join(root_path, "images_result", filename),model_character))
    print(list_character_first)
    print(list_character_second)

else:
    candidates.sort(key=lambda item: item[0])
    candidates = [item[2] for item in candidates]     
    for character in candidates:
        i+=1
        filename = f"region_{i}.jpg"
        cv2.imwrite(os.path.join(root_path, "images_result", filename), character)
        list_character.append(predict_image(os.path.join(root_path, "images_result", filename), model_character))
    print(list_character)