import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

        self.two_var_s =  2 * sigma_s**2
        self.two_var_r =  2 * sigma_r**2
        self.spatial_kernel = self.spatial_kernel_2d()
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        padded_guidance = padded_guidance.astype(np.float64) / 255.0
        guidance = guidance.astype(np.float64) / 255.0
        output = np.zeros_like(img)
        range_kernel_f = self.range_kernel_3d if guidance.ndim == 3 else self.range_kernel_1d


        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # Extract window
                window = padded_img[i:i+self.wndw_size, j:j+self.wndw_size, :]
                window_guidance = padded_guidance[i:i+self.wndw_size, j:j+self.wndw_size]

                # Compute range kernel
                range_kernel = range_kernel_f(window_guidance, guidance[i, j])

                # Compute joint bilateral filter
                joint_bilateral_filter = self.spatial_kernel * range_kernel

                # Compute output
                output[i, j, :] = np.sum(window * joint_bilateral_filter[:, :, np.newaxis], axis=(0, 1)) / np.sum(joint_bilateral_filter)


        return np.clip(output, 0, 255).astype(np.uint8)
    
    def spatial_kernel_2d(self):
        spatial_kernel = np.zeros((self.wndw_size, self.wndw_size))
        center_i = self.wndw_size // 2
        center_j = self.wndw_size // 2

        for x in range(self.wndw_size):
            for y in range(self.wndw_size):
                spatial_kernel[x, y] = np.exp(-((x - center_i)**2 + (y - center_j)**2) / self.two_var_s)

        return spatial_kernel
    
    def range_kernel_1d(self, window_guidance, guidance):
        """Compute range kernel for 1D guidance image
        Args:
            window_guidance (np.ndarray): Window of guidance image. Shape: (window_size, window_size)
            guidance (float): Guidance value at (xp, yp). Shape: (1, )
        
        Returns:
            range_kernel (np.ndarray): Range kernel
        """
        range_kernel = np.exp(-((window_guidance - guidance)**2) / self.two_var_r)
        return range_kernel
    
    def range_kernel_3d(self, window_guidance, guidance):
        """Compute range kernel for 3D guidance image
        Args:
            window_guidance (np.ndarray): Window of guidance image. Shape: (window_size, window_size, 3)
            guidance (np.ndarray): Guidance value at (xp, yp). Shape: (1, 1, 3)
            
        Returns:
            range_kernel (np.ndarray): Range kernel
        """

        range_kernel = np.exp(-((window_guidance - guidance)**2).sum(axis=2) / self.two_var_r)
        return range_kernel
    