import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def ae(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(output.astype(np.int32) - target.astype(np.int32)))

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    # Read image
    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    print(f'Image: {args.image_path}\n')

    # Create output directory
    if not os.path.exists('./output'):
        os.mkdir('./output')

    # Read setting file
    with open(args.setting_path, 'r') as f:
        lines = f.readlines()

    rgb_params = ['cv2.COLOR_BGR2GRAY']
    for line in lines[1:-1]:
        rgb_coeff = line.split(',')
        rgb_params.append((float(rgb_coeff[0]), float(rgb_coeff[1]), float(rgb_coeff[2])))
    
    sigma_s = int(lines[-1].split(',')[1])
    sigma_r = float(lines[-1].split(',')[3])

    # Generate gray-scale images
    gray_imgs = [img_gray]
    for rgb_param in rgb_params[1:]:
        gray_img = img_rgb[:, :, 0] * rgb_param[0] + img_rgb[:, :, 1] * rgb_param[1] + img_rgb[:, :, 2] * rgb_param[2]
        gray_imgs.append(gray_img)

    # Create JBF class
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)

    errors = []
    for i, gray_img in enumerate(gray_imgs):
        jbf_out = JBF.joint_bilateral_filter(img_rgb, gray_img).astype(np.uint8)
        cv2.imwrite(f'./output/{args.image_path[-5]}_grayscale_{i + 1}.png', gray_img)
        cv2.imwrite(f'./output/{args.image_path[-5]}_filtered_rgb_{i + 1}.png', cv2.cvtColor(jbf_out, cv2.COLOR_RGB2BGR))
        errors.append(ae(bf_out, jbf_out))
        print(f'Guidance {i + 1}\t(R, G, B) = {rgb_params[i]},\tError: {errors[-1]}')

    # Find the best and worst setting
    best_setting = np.argmin(errors)
    worst_setting = np.argmax(errors)
    print(f'\nBest setting: (R, G, B) = {rgb_params[best_setting]},\tError: {errors[best_setting]}')
    print(f'Worst setting: (R, G, B) = {rgb_params[worst_setting]},\tError: {errors[worst_setting]}')    


if __name__ == '__main__':
    main()