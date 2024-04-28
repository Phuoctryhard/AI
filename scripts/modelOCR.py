import cv2
import numpy as np
from ultralytics import YOLO
from keras import models

def predict_image(image_path, model):
    img = cv2.imread(image_path)  
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (40,30))
    img_array = np.array(resized_img) 
    img_array=img_array/255.0
    img_input = img_array.reshape((-1, 30, 40, 1))
    pred = model.predict(img_input)
    # labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    predicted_label = labels[np.argmax(pred)]
    return predicted_label

model = YOLO("best.pt")
model_character = models.load_model('model_license_plate_v9.h5')
img = cv2.imread("Image/437167569_2141178052923210_7457484411368920934_n.png")
# img_crop = img
# X,Y = 0,0
# W,H,_= img_crop.shape
results = model.predict(img)
X, Y, W, H = None, None, None, None
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        X = box.xyxy[0][0]  
        Y = box.xyxy[0][1]
        W = box.xywh[0][2]
        H = box.xywh[0][3]
img_crop = img[int(Y)-2: int(Y)+int(H)+2, int(X)-2: int(X)+int(W)+2]
cv2.imwrite("cropped_image.jpg", img_crop)
cv2.imshow("Crop_img", img_crop)
cv2.waitKey()

img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

cv2.imshow("blur", blur)
cv2.waitKey()
binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, int(W/10) if int(W/10)%2 !=0  else int(W/10)+1, 10)
# _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow("binary", binary)
cv2.waitKey()
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

        # Kiểm tra các điều kiện để loại bỏ nhiễu như dâu "." hay "-"
        if 0.2 < aspect_ratio < 0.8 and solidity > 0.1 and 0.2 < height_ratio < 1.0:
            bounding_rects.append((x, y, w, h))
            # Trích xuất ký tự
            character = img_crop[y-3: y + h+3, x-int(h*13/40-w/2):x + w+int(h*13/40-w/2)]
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


    for idx, (x, y,char)  in enumerate(first_lines):
    # Vẽ hình chữ nhật bao quanh ký tự
        cv2.rectangle(img, (x+int(X)-4, y+int(Y)-3), (x+int(X) + char.shape[1] - 6, y+int(Y) + char.shape[0] - 5), (0, 255, 0), 2)
    
    # Hiển thị ký tự lên ảnh
        cv2.putText(img, list_character_first[idx], (x+int(X)-3, y + int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   
                
    for second_line in second_lines:
            i+=1
            filename = f"region_{i}.jpg"
            cv2.imwrite(filename, second_line[2])
            list_character_second.append(predict_image(filename,model_character))
    print(list_character_first)
    print(list_character_second)

    for idx, (x, y,char)  in enumerate(second_lines):
    # Vẽ hình chữ nhật bao quanh ký tự
        cv2.rectangle(img, (x+int(X)-4, y+int(Y)-3), (x+int(X) + char.shape[1] - 6, y+int(Y) + char.shape[0] - 5), (0, 255, 0), 2)
    
    # Hiển thị ký tự lên ảnh
        cv2.putText(img, list_character_second[idx], (x+int(X)-3, y + int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Hiển thị ảnh với các ký tự đã nhận dạng được vẽ lên
    cv2.imshow("Detected Characters", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
else:
    candidates.sort(key=lambda item: item[0])
    candidates = [item for item in candidates]     
    for character in candidates:
            i+=1
            filename = f"region_{i}.jpg"
            cv2.imwrite(filename, character[2])
            list_character.append(predict_image(filename,model_character))

    print(list_character)
    for idx, (x, y,char)  in enumerate(candidates):
    # Vẽ hình chữ nhật bao quanh ký tự
        cv2.rectangle(img, (x+int(X)-4, y+int(Y)-3), (x+int(X) + char.shape[1] - 6, y+int(Y) + char.shape[0] - 5), (0, 255, 0), 2)
    
    # Hiển thị ký tự lên ảnh
        cv2.putText(img, list_character[idx], (x+int(X)-3, y + int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Hiển thị ảnh với các ký tự đã nhận dạng được vẽ lên
    cv2.imshow("Detected Characters", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
