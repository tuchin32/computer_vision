import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def orb_bfmatching(img_query, img_train):
    '''
        Brute-Force Matcher with ORB descriptors
        :param img_query:   queryImage
        :param img_train:   trainImage
        :return:            matched points in queryImage and trainImage
    '''

    # initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp_query, des_query = orb.detectAndCompute(img_query, None)
    kp_train, des_train = orb.detectAndCompute(img_train, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # match descriptors
    matches = bf.match(des_query, des_train)

    # sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)

    # find good matched points
    pts_query, pts_train = [], []
    for m in matches:
        pts_query.append(kp_query[m.queryIdx].pt)
        pts_train.append(kp_train[m.trainIdx].pt)
    pts_query, pts_train = np.asarray(pts_query), np.asarray(pts_train)
    return pts_query, pts_train

def projection(pts_src, H):
    """
        Project points from source image to destination image
        :param pts_src:     points in source image
        :param H:           homography matrix
        :return:            points in destination image
    """

    homo_pts_src = np.vstack((pts_src.T, np.ones((1, pts_src.shape[0]))))    # (3, N)
    homo_pts_dst = np.dot(H, homo_pts_src)    # (3, N)
    homo_pts_dst /= homo_pts_dst[-1, :]
    pts_dst = (homo_pts_dst.T)[:, :-1]  # (N, 2)
    return pts_dst


def ransac(pts_src, pts_dst, threshold=3, max_iter=1000):
    """
        RANSAC algorithm
        :param pts_src:     points in source image
        :param pts_dst:     points in destination image
        :param threshold:   threshold for inlier
        :param max_iter:    maximum iteration
        :return:            best homography matrix H
    """

    best_inliers = 0
    best_H = None

    for _ in range(max_iter):
        # randomly choose 4 points to compute homography
        indices = random.sample(range(pts_src.shape[0]), 4)
        H = solve_homography(pts_src[indices], pts_dst[indices])

        # count number of inliers whose distance is small than threshold
        distance = np.linalg.norm(pts_dst - projection(pts_src, H), axis=1)
        inliers = np.sum(distance < threshold)

        # update best homography
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H
    
    return best_H

def alpha_blend(img1, img2):
    """
        Alpha blending for two images
        :param img1:    image 1
        :param img2:    image 2
        :return:        blended image
    """
    assert img1.shape == img2.shape

    # find the overlapping region and the corresponding coordinates
    overlap = np.logical_and(np.sum(img1, axis=2) != 0, np.sum(img2, axis=2) != 0).astype(np.uint8)
    x_min = np.min(np.where(overlap == 1)[1])
    x_max = np.max(np.where(overlap == 1)[1])
    x_min += (x_max - x_min) // 4   # only blend the middle part
    x_max -= (x_max - x_min) // 4

    # create mask for alpha blending
    mask = (np.copy(overlap)).astype(np.float32)    # (H, W)
    mask[:, :x_min] *= 1.0  # left part
    mask[:, x_min:x_max] *= (1.0 - np.arange(x_max - x_min) / (x_max - x_min))  # middle part
    mask[:, x_max:] *= 0.0  # right part
    mask = np.stack([mask] * 3, axis=2) # (H, W, 3)

    # alpha blending
    blend = np.zeros_like(img1)
    blend[overlap == 1] = img1[overlap == 1] * mask[overlap == 1] + img2[overlap == 1] * (1 - mask[overlap == 1])
    blend[overlap == 0] = img1[overlap == 0] + img2[overlap == 0]
    return blend

def panorama(imgs):
    """
        Image stitching with estimated homograpy between consecutive
        :param imgs:    list of images to be stitched
        :return:        stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1. feature detection & matching
        pts1, pts2 = orb_bfmatching(im1, im2)
        assert pts1.shape[0] == pts2.shape[0]

        # TODO: 2. apply RANSAC to choose best H
        H = ransac(pts2, pts1, threshold=3, max_iter=1000)

        # TODO: 3. chain the homographies
        H_chain = np.dot(last_best_H, H)

        # TODO: 4. apply warping
        out = warping(im2, np.zeros_like(dst), H_chain, 0, h_max, 0, w_max, direction='b')

        # BONUS: 5. alpha blending
        out = alpha_blend(dst, out)

        # update the stitched canvas
        dst = np.copy(out)
        last_best_H = np.copy(H_chain)

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)