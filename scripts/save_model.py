import os
MAIN_FOLDER_PATH = os.getcwd()
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
export_main_v2_path = MAIN_FOLDER_PATH + "\\" + "scripts\\exporter_main_v2.py"

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

command_export = f"python {export_main_v2_path} --input_type image_tensor --pipeline_config_path {files['PIPELINE_CONFIG']} --trained_checkpoint_dir {paths['CHECKPOINT_PATH']} --output_directory {paths['EXPORTED_MODEL_PATH']}"
# command_export = f"python {export_main_v2_path} --input_type image_tensor --pipeline_config_path {files['PIPELINE_CONFIG']} --trained_checkpoint_dir {r"D:\Code_school_nam3ki2\TestModel\model\detect_liscense_plate"} --output_directory {r'D:\Code_school_nam3ki2\TestModel\check'}"

os.system(command_export)