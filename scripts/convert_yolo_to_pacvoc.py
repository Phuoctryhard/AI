import xml.etree.cElementTree as ET
import cv2
import os

img_dir = r"D:\biensoxe.v1i.yolov7pytorch\collect\images"
annotation_dir = r"D:\biensoxe.v1i.yolov7pytorch\collect\labels"
save_dir = r"D:\biensoxe.v1i.yolov7pytorch\collect\labels_voc"
# Read image
def ReadImage(filename):
    img = cv2.imread(filename)
    return img

def ReadAnnotation(filename):
    with open(filename, 'r') as f:
        data = f.readline().split()
        data = [float(i) for i in data]
    return data


def convert_yolo_to_xml(yolo_data_path, img_path):
    yolo_data = ReadAnnotation(yolo_data_path)
    img = ReadImage(img_path)
    img_height,img_width, _ = img.shape
    img_name = img_path.split('\\')[-1]
    # Convert YOLO center coordinates (normalized) to XML format
    xmin = (yolo_data[1] - yolo_data[3]/2.) * img_width
    xmax = (yolo_data[1] + yolo_data[3]/2.) * img_width
    ymin = (yolo_data[2] - yolo_data[4]/2.) * img_height
    ymax = (yolo_data[2] + yolo_data[4]/2.) * img_height

    # Create XML tree structure
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = img_name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"
    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = "licence"
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "occluded").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(int(xmin))
    ET.SubElement(bndbox, "ymin").text = str(int(ymin))
    ET.SubElement(bndbox, "xmax").text = str(int(xmax))
    ET.SubElement(bndbox, "ymax").text = str(int(ymax))

    # Convert to string
    xml_str = ET.tostring(annotation)

    # Save to file
    save_path = os.path.join(save_dir, img_name.split('.')[0] + ".xml")
    with open(save_path, "wb") as f:
        f.write(xml_str)