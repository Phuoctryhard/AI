import os
import object_detection
import tensorflow as tf
from object_detection.utils import config_util
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(tf.config.list_physical_devices('GPU'))
print('Using device:', device)

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
MAIN_FOLDER_PATH = os.getcwd()

paths = {
    'WORKSPACE_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join(MAIN_FOLDER_PATH, 'Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join(MAIN_FOLDER_PATH,'Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join(MAIN_FOLDER_PATH,'Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'EXPORTED_MODEL_PATH': os.path.join(MAIN_FOLDER_PATH,'Tensorflow', 'workspace','exported-models', CUSTOM_MODEL_NAME),
    'PROTOC_PATH':os.path.join(MAIN_FOLDER_PATH,'Tensorflow','protoc')
}

files = {
    'PIPELINE_CONFIG':os.path.join(MAIN_FOLDER_PATH,'Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Create Directory Structure
for path in paths.values():
    if not os.path.exists(path):
        os.makedirs(path)

# Create Label Map
labels = [
    {'name':'licence', 'id':1}
]   

# Create TF records
if not os.path.exists(files['TF_RECORD_SCRIPT']):
    os.system(f"git clone https://github.com/nicknochnack/GenerateTFRecord {paths['SCRIPTS_PATH']}")
os.system(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'train')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'train.record')}")
os.system(f"python {files['TF_RECORD_SCRIPT']} -x {os.path.join(paths['IMAGE_PATH'], 'test')} -l {files['LABELMAP']} -o {os.path.join(paths['ANNOTATION_PATH'], 'test.record')}")

#Copy model config từ file config của pretrain_model
command_copy_model_config = f"cp {os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config')} {os.path.join(paths['CHECKPOINT_PATH'])}"
os.system(command_copy_model_config)

# Đọc tệp pipeline.config và lấy ra tất cả các cấu hình.
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

# Custom lại model
configs['model'].ssd.num_classes = len(labels)
configs['train_config'].batch_size = 4
configs['train_config'].fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
configs['train_config'].fine_tune_checkpoint_type = "detection"
configs['train_input_config'].label_map_path = files['LABELMAP']
configs['train_input_config'].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
configs['eval_input_configs'][0].label_map_path = files['LABELMAP']
configs['eval_input_configs'][0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

# Lưu lại custom model vào file config
pipeline_config = config_util.create_pipeline_proto_from_configs(configs)
config_util.save_pipeline_config(pipeline_config, os.path.dirname(files['PIPELINE_CONFIG']))
    
#Train model
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=10000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'], files['PIPELINE_CONFIG'])

os.system(command)