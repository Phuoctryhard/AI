import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import os
import sys

# Kiểm tra xem có đối số dòng lệnh nào được truyền vào không
image_path = ""
save_img_path = ""
if len(sys.argv) > 2:
    image_path = sys.argv[1]
    save_img_path = sys.argv[2]
else:
    print("No image path is provided")
    sys.exit(1)

if not os.path.exists(image_path) or save_img_path == "":
    print("Image path does not exist  or save image path is not provided")
    sys.exit(1)
    
project_dir = os.getcwd()

model_path = os.path.join(project_dir, 'model/improve_image_resolution/models/RRDB_ESRGAN_x4.pth')  
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu

test_img_path = image_path
# test_img_folder = os.path.join(project_dir, 'images_result/*')

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))
img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
img = img * 1.0 / 255
img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
img_LR = img.unsqueeze(0)
img_LR = img_LR.to(device)

with torch.no_grad():
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round()
cv2.imwrite(save_img_path, output)
print(f"Saving image to {save_img_path} successfully")


# idx = 0
# for path in glob.glob(test_img_folder):
#     idx += 1
#     base = osp.splitext(osp.basename(path))[0]
#     print(idx, base)
#     # read images
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     img = img * 1.0 / 255
#     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#     img_LR = img.unsqueeze(0)
#     img_LR = img_LR.to(device)

#     with torch.no_grad():
#         output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#     output = (output * 255.0).round()
#     cv2.imwrite(os.path.join(project_dir, 'results/{:s}.jpg').format(base), output)
