import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

        self.gaussian_images = None
        self.dog_images = None

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        for oct in range(self.num_octaves):
            oct_images = []
            # Downsample the image by half
            if oct == 0:
                oct_images.append(image)
            else:
                oct_images.append(cv2.resize(gaussian_images[-1][-1], (0, 0), fx=0.5**oct, fy=0.5**oct, 
                                             interpolation=cv2.INTER_NEAREST))

            # Implement the Gaussian filter with different sigma values
            base_image = oct_images[0].copy()
            for i in range(1, self.num_guassian_images_per_octave):
                oct_images.append(cv2.GaussianBlur(base_image, (0, 0), self.sigma**i))
            gaussian_images.append(oct_images)
        self.gaussian_images = gaussian_images


        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for oct in range(self.num_octaves):
            oct_images = []
            for i in range(self.num_DoG_images_per_octave):
                oct_images.append(cv2.subtract(gaussian_images[oct][i + 1], gaussian_images[oct][i]))
            dog_images.append(oct_images)


        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for oct in range(self.num_octaves):
            # dog_img: (4, h, w)
            dog_img = np.array(dog_images[oct])
            
            # Thresholding the value
            dog_img[np.abs(dog_img) < self.threshold] = 0
            
            for i in range(1, self.num_DoG_images_per_octave - 1):
    
                # Find local extremum
                for y in range(1, dog_img.shape[1] - 1):
                    for x in range(1, dog_img.shape[2] - 1):
                        if dog_img[i, y, x] != 0:
                            # Check if the pixel is a local maximun or local minimum
                            if np.max(dog_img[i - 1:i + 2, y - 1:y + 2, x - 1:x + 2]) == dog_img[i, y, x] \
                                or np.min(dog_img[i - 1:i + 2, y - 1:y + 2, x - 1:x + 2]) == dog_img[i, y, x]:
                                keypoints.append((y * (oct + 1), x * (oct + 1)))

            dog_images[oct] = dog_img

        self.dog_images = dog_images


        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
