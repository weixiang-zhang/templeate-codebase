import os
import glob
import cv2 as cv


# input_folder = "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/visual_06_compare"
input_folder = "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/visual_backbone_023_kodak"
output_path = os.path.join(input_folder, "output")

input_path_list = glob.glob(os.path.join(input_folder, '*'))
print('input_path_list: ', input_path_list)

os.makedirs(output_path, exist_ok=True)

def crop_and_rec_image(img_path,output_path, key="test"):
    img = cv.imread(img_path) # h,w,c
    print('img: ', img)

    # crop image
    left, right, upper,lower = [134,268,73,207] # 06
    left, right, upper,lower = [467, 586,135, 216] # kodak 23
    cropped_img = img[upper:lower, left:right]
    cv.imwrite(os.path.join(output_path, f"{key}_crop.png"), cropped_img)
    
    # rectangle
    color = (255, 0, 0)  # 红色 (B, G, R)
    thickness = 2  # 框的厚度
    cv.rectangle(img, (left, upper), (right, lower), color, thickness)
    cv.imwrite(os.path.join(output_path, f"{key}_all.png"), img)

for input_path in input_path_list:
    filename= os.path.splitext(os.path.basename(input_path))[0]
    print('filename: ', filename)
    print('input_path: ', input_path)

    
    crop_and_rec_image(input_path, output_path, filename)


# crop_and_rec_image(img_path="/home/wxzhang/projects/coding4paper/projects/subband/data/Kodak/kodim17.png",
#                    output_path=output_path,
#                    key="gt")