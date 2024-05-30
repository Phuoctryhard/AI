import os
from PIL import Image
import io
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import time, progressbar
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.core import data_parser
from object_detection.core import standard_fields as fields
import matplotlib.pyplot as plt
from object_detection.metrics.tf_example_parser import BoundingBoxParser, StringParser, Int64Parser, FloatParser
import seaborn as sns

tf.compat.v1.flags.DEFINE_string('input_tfrecord_path', None,
                                 'Input tf record path')
tf.compat.v1.flags.DEFINE_string('output_path', None,
                                 'Path to the output CSV.')
tf.compat.v1.flags.DEFINE_string('inference_graph', None,
                                 'Path to the inference graph with embedded weights.')
tf.compat.v1.flags.DEFINE_string('class_labels', None,
                                 'Path to classes.pbtxt file.')
tf.compat.v1.flags.DEFINE_string('draw_option', "False",
                                 'Whether or not to save images with boxes drawn')
tf.compat.v1.flags.DEFINE_string('draw_save_path', None,
                                 'If using draw_option, where to save images')

FLAGS = tf.compat.v1.flags.FLAGS
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5

class CustomParser(data_parser.DataToNumpyParser):

  def __init__(self):
    self.items_to_handlers = {
        fields.InputDataFields.image: StringParser(
                    fields.TfExampleFields.image_encoded),
        fields.InputDataFields.groundtruth_boxes: BoundingBoxParser(
                    fields.TfExampleFields.object_bbox_xmin,
                    fields.TfExampleFields.object_bbox_ymin,
                    fields.TfExampleFields.object_bbox_xmax,
                    fields.TfExampleFields.object_bbox_ymax),
        fields.InputDataFields.groundtruth_classes: Int64Parser(
                    fields.TfExampleFields.object_class_label)
    }
    # Optional
    self.filename = StringParser(fields.TfExampleFields.filename)

  def parse(self, tf_example):
    results_dict = {}
    parsed = True
    for key, parser in self.items_to_handlers.items():
        results_dict[key] = parser.parse(tf_example)
        parsed &= (results_dict[key] is not None)

    # TODO: need to test
    filename = self.filename.parse(tf_example)
    results_dict['filename'] = filename # could be None

    return results_dict if parsed else None

def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())
    
    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def process_detections(input_tfrecord_path, model, categories, draw_option, draw_save_path):
    data_parser = CustomParser()
    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))
    
    # Create dataset from records
    input_dataset = tf.data.TFRecordDataset(input_tfrecord_path)

    with progressbar.ProgressBar(max_value=progressbar.UnknownLength) as bar:
        for image_index, record in enumerate(input_dataset):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            decoded_dict = data_parser.parse(example)
            
            if decoded_dict:
                image               = decoded_dict[fields.InputDataFields.image]
                image = Image.open(io.BytesIO(image)).convert('RGB')  # Convert image to RGB
                input_tensor = np.expand_dims(image, axis=0)
                groundtruth_boxes   = decoded_dict[fields.InputDataFields.groundtruth_boxes]
                groundtruth_classes = decoded_dict[fields.InputDataFields.groundtruth_classes].astype('uint8')
                detections          = model(input_tensor) # Run model inference
                detection_scores    = detections['detection_scores'][0].numpy()
                detection_boxes     = detections['detection_boxes'][0].numpy()[detection_scores >= CONFIDENCE_THRESHOLD]
                detection_classes   = detections['detection_classes'][0].numpy()[detection_scores >= CONFIDENCE_THRESHOLD].astype('uint8')
                filename            = decoded_dict[fields.InputDataFields.filename]
                filename            = filename.decode('UTF-8') if filename is not None else f'image-{image_index}.png'

                matches = []
                image_index += 1
                
                for i, groundtruth_box in enumerate(groundtruth_boxes):
                    for j, detection_box in enumerate(detection_boxes):
                        iou = compute_iou(groundtruth_box, detection_box)
                        if iou > IOU_THRESHOLD:
                            matches.append([i, j, iou])

                matches = np.array(matches)
                if matches.shape[0] > 0:
                    # Sort list of matches by descending IOU so we can remove duplicate detections
                    # while keeping the highest IOU entry.
                    matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                    
                    # Remove duplicate detections from the list.
                    matches = matches[np.unique(matches[:,1], return_index=True)[1]]
                    
                    # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                    # our previous sort.
                    matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]
                    
                    # Remove duplicate ground truths from the list.
                    matches = matches[np.unique(matches[:,0], return_index=True)[1]]
                    
                for i in range(len(groundtruth_boxes)):
                    if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
                        confusion_matrix[groundtruth_classes[i] - 1][int(detection_classes[int(matches[matches[:,0] == i, 1][0])] - 1)] += 1 
                    else:
                        confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
                        
                for i in range(len(detection_boxes)):
                    if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
                        confusion_matrix[confusion_matrix.shape[0] - 1][int(detection_classes[i] - 1)] += 1
                
                if draw_option:
                    draw(filename, draw_save_path, input_tensor[0], categories,
                        groundtruth_boxes, groundtruth_classes, detection_boxes, detection_classes, detection_scores)
            else:
                print(f'Skipping image {image_index}')

            bar.update(image_index)

    print(f'Processed {image_index + 1} images')
    # Lưu ma trận vào file với định dạng là số nguyên
    np.savetxt("confusion_matrix.csv", confusion_matrix, fmt='%i', delimiter=",")
    return confusion_matrix
    
def display(confusion_matrix, categories, output_path):
    results = []

    for i in range(len(categories)):
        id = categories[i]['id'] - 1
        name = categories[i]['name']
        
        total_target = np.sum(confusion_matrix[id,:])
        total_predicted = np.sum(confusion_matrix[:,id])
        
        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)
        
        results.append({'category' : name, f'precision_@{IOU_THRESHOLD}IOU' : precision, f'recall_@{IOU_THRESHOLD}IOU' : recall})
    
    df = pd.DataFrame(results)
    accuracies = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Tính toán F1 Score
    df['f1_score_@0.5IOU'] = 2 * (df['precision_@0.5IOU'] * df['recall_@0.5IOU']) / (df['precision_@0.5IOU'] + df['recall_@0.5IOU'])
    # Vẽ biểu đồ
    ax1.set_title('Biểu đồ Precision, Recall, F1 Score @0.5IOU')
    ax1 = df.plot(y=['precision_@0.5IOU', 'recall_@0.5IOU', 'f1_score_@0.5IOU'], kind='bar', figsize=(10,7),
                  ax=ax1, label=['precision_@0.5IOU', 'recall_@0.5IOU', 'f1_score_@0.5IOU'], width=0.7)
    # Xoay nhãn về thẳng
    ax1.set_xticklabels(df['category'], rotation=0)
    ax1.set_ylabel('Giá trị', rotation=0, labelpad=30)
    # Hiện giá trị các nhãn trên cột và canh giữa
    for p in ax1.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 
        ax1.annotate(f'{height:.2f}', (x + width/2, y + height*1.02), ha='center')
    ax1.legend(loc='lower right')
    # Calculate accuracy
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    ax2.set_title('Model Accuracy')
    # Plot 'total_accuracy'
    ax2.bar('accuracy', accuracy, color='blue', label='accuracy', width=0.7)
    ax2.set_xlim([-1, 1]) 
    for p in ax2.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy() 
        ax2.annotate(f'{height:.2f}', (x + width/2, y + height*1.02), ha='center')
    ax2.legend(loc='lower right')
    plt.savefig('Precision_Recall_f1_score_Accuracy.png')
    df.to_csv(output_path)

def draw(image_name, image_path, image, categories, groundtruth_boxes, groundtruth_classes, detection_boxes, detection_classes, detection_scores):
    # Convert category array to category index
    cat_index = {}
    for i, item in enumerate(categories):
        cat_index[i+1] = item
    image_viz = image.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_viz,
            groundtruth_boxes,
            groundtruth_classes,
            None, # scores = None triggers ground truth mode
            cat_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=CONFIDENCE_THRESHOLD,
            agnostic_mode=False,
            line_thickness=3,
            groundtruth_box_visualization_color='blue') # opencv will make this red
    # Draw detections
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_viz,
            detection_boxes,
            detection_classes,
            detection_scores,
            cat_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=CONFIDENCE_THRESHOLD,
            agnostic_mode=False,
            line_thickness=1)
    image_viz = cv2.cvtColor(image_viz, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(image_path, image_name), image_viz)

def plot_confusion_matrix(cm, classes, cmap=plt.cm.Blues):
    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(10,7))
    ax = sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = cmap)
    ax.set_xticklabels(classes, rotation=0, va='top', position=(0,1.04))
    ax.tick_params(axis='x', labeltop=True, top=True, labelbottom=False, bottom=False)
    ax.set_yticklabels(classes, rotation=0, va='top')
    ax.tick_params(axis='y', labeltop=True, top=True, labelbottom=False, bottom=False)

    ax.yaxis.set_label_position("left") # Di chuyển nhãn y sang trái
    #Xoay nhãn y 90 độ
    ax.set_ylabel('Nhãn thực tế', rotation=0, labelpad=30)
    ax.set_xlabel('Nhãn dự đoán', rotation=0, labelpad=30)
    ax.set_title('Ma trận nhầm lẫn', size = 15, pad=20)
    plt.savefig('confusion_matrix.png')

def main(_):
    required_flags = ['input_tfrecord_path', 'output_path',
                      'inference_graph', 'class_labels']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError(f'Flag --{flag_name} is required')
    
    input_tfrecord_path = FLAGS.input_tfrecord_path
    print(input_tfrecord_path)

    draw_option    = FLAGS.draw_option.lower() == 'true'
    draw_save_path = FLAGS.draw_save_path

    if draw_option:
        if draw_save_path is None:
            raise ValueError('If --draw_option is True, --draw_save_path is required')

    # Get class names
    label_map  = label_map_util.load_labelmap(FLAGS.class_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)

    # Load model
    print('Loading model...')
    model = tf.saved_model.load(FLAGS.inference_graph)

    # Run inference and compute confusion matrix
    print('Evaluating model...')
    confusion_matrix = process_detections(input_tfrecord_path, model, categories, draw_option, draw_save_path)
    classes = [category['name'] for category in categories] + ['No detection']
    
    # Save to CSV
    print('Saving confusion matrix...')
    display(confusion_matrix, categories, FLAGS.output_path)
    plot_confusion_matrix(confusion_matrix, classes)
    print('Done!')  

if __name__ == '__main__':
    tf.compat.v1.app.run()
