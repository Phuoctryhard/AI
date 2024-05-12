from pylabel import importer

path_to_annotations = "path/to/annotations"
path_to_images = "path/to/images"
save_dir = "save_dir"
yolo_class = ['licence']

dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=yolo_class, img_ext='jpg', name="coco128")

#export dataset
dataset.export.ExportToVoc(output_path=save_dir)