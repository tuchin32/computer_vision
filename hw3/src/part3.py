import numpy as np
import cv2
from utils import solve_homography, warping

def distortion(points, k1=0, k2=0, p1=0, p2=0, s1=0, s2=0):
    x, y = points[:, 0], points[:, 1]
    r_square = x ** 2 + y ** 2
    k = 1 + k1 * r_square + k2 * (r_square ** 2)
    u = x * k + 2 * p2 * x * y + p1 * (r_square + 2 * (x ** 2)) + s1 * r_square
    v = y * k + p2 * (r_square + 2 * (y ** 2)) + 2* p1 * x * y + s2 * r_square
    new_points = np.array([u, v]).T
    return new_points

if __name__ == '__main__':

    # ================== Part 3 ========================
    secret1 = cv2.imread('../resource/BL_secret1.png')
    secret2 = cv2.imread('../resource/BL_secret2.png')
    corners1 = np.array([[429, 337], [517, 314], [570, 361], [488, 380]])
    corners2 = np.array([[346, 196], [437, 161], [483, 198], [397, 229]])
    h, w, c = (500, 500, 3)
    dst = np.zeros((h, w, c))
    

    # TODO: call solve_homography() & warping
    output3_1 = None
    output3_2 = None
    dst_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    H3_1 = solve_homography(corners1, dst_corners)
    output3_1 = warping(secret1, np.copy(dst), H3_1, 0, h, 0, w, direction='b')
    cv2.imwrite('output3_1.png', output3_1)

    H3_2 = solve_homography(corners2, dst_corners)
    output3_2 = warping(secret2, np.copy(dst), H3_2, 0, h, 0, w, direction='b')
    cv2.imwrite('output3_2.png', output3_2)