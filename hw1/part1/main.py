import os
import cv2
import argparse
import numpy as np
from DoG import Difference_of_Gaussian

def plot_dog(dog, save_path='./output/'):
    dog_images = dog.dog_images

    for i in range(dog.num_octaves):
        for j in range(dog.num_DoG_images_per_octave):
            image = dog_images[i][j].copy()
            image = (image - image.min()) / (image.max() - image.min()) * 255
            cv2.imwrite(save_path + f'DoG{i + 1}-{j + 1}.png', image)

def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float64)

    ### TODO ###
    DoG = Difference_of_Gaussian(args.threshold)
    keypoints = DoG.get_keypoints(img)
    print(f'Total number of keypoints: {len(keypoints)}')

    # save keypoints
    if not os.path.exists('./output'):
        os.mkdir('./output')

    plot_keypoints(img, keypoints, f'./output/{args.image_path[-5]}_thres{int(args.threshold)}.png')
    plot_dog(DoG, save_path='./output/')


if __name__ == '__main__':
    main()