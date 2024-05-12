import os
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


# if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
#     os.system(f"git clone https://github.com/tensorflow/models {os.path.join(paths['APIMODEL_PATH'])}")



# import wget
# if len(os.listdir(paths['PROTOC_PATH'])) == 0:
#     url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
#     wget.download(url)
#     os.system(f"move protoc-3.15.6-win64.zip {paths['PROTOC_PATH']}")
#     os.system(f"cd {paths['PROTOC_PATH']} && tar -xf protoc-3.15.6-win64.zip")
# os.environ['PATH'] += os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   


# # Compile protobuf files and install all required libraries
# os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install")

# os.system("cd Tensorflow/models/research/slim && pip install -e .")

# Verify Installation
# VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# os.system(f"python {VERIFICATION_SCRIPT}")