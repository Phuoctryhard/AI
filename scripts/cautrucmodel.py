import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import numpy as np
import pandas as pd

config_path = r'model\detect_license_plate\pipeline.config'
label_path = r'config\label_map.pbtxt'
checkpoint_path = r'model\detect_license_plate\checkpoint\ckpt-0'
MAIN_FOLDER_PATH = os.getcwd()

category_index = label_map_util.create_category_index_from_labelmap(os.path.join(MAIN_FOLDER_PATH, label_path))
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(os.path.join(MAIN_FOLDER_PATH, config_path))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(MAIN_FOLDER_PATH, checkpoint_path)).expect_partial()

dummy_input = tf.random.uniform((1, 320, 320, 3))

dummy_output = detection_model(dummy_input)
total_params = 0
total_train_params = 0
total_untrain_params = 0
layer_cnt = 0
list_layer = []
list_input_shape = []
list_param = []
for var in detection_model.variables:
    layer_cnt += 1
    num_params = np.prod(var.shape)
    total_params += num_params
    if var.trainable:
        total_train_params += num_params
    else:
        total_untrain_params += num_params
    list_layer.append(var.name)
    list_input_shape.append(var.shape)
    list_param.append(num_params)

data = {
    "Layer": list_layer,
    "Input Shape": list_input_shape,
    "Parameters": list_param
}
df = pd.DataFrame(data)

pd.set_option('display.max_rows', None)

# Print the table
print(df.to_string(index=False))
print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("Total Layers: ", layer_cnt)
print("Total Parameters: ", total_params)
print("Total Trainable Parameters: ", total_train_params)
print("Total Untrainable Parameters: ", total_untrain_params)
