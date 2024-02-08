import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # 0. Transform u, v to homogeneous coordinate
    u = np.hstack((u, np.ones((N, 1))))
    v = np.hstack((v, np.ones((N, 1))))
    
    # TODO: 1.forming A
    A = np.zeros((2 * N, 9))
    A[::2, :3] = u
    A[::2, 3:6] = 0
    A[::2, 6:] = -u * v[:, 0].reshape((-1, 1))
    A[1::2, :3] = 0
    A[1::2, 3:6] = u
    A[1::2, 6:] = -u * v[:, 1].reshape((-1, 1))

    # TODO: 2.solve H with A
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape((3, 3))
    # H /= H[-1, -1]

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    pass

    if direction == 'b':
        '''
        backward warping
            :var interpolation:     'bilinear' or 'nearest'
            :var warped_homo_coord: (3, N). N equals to (ymax-ymin) * (xmax-xmin)
            :var homo_coord:        (3, N)
            :var coord:             (N, 2)
            :var mask:              (N,)
            :var filtered_coord     (M, 2). M is the number of elements passing the mask.
        '''
        interpolation = 'bilinear'
        warped_homo_coord = np.vstack((x.flatten(), y.flatten(), np.ones((1, x.size))))     # (3, N)

        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        homo_coord = np.dot(H_inv, warped_homo_coord)
        homo_coord /= homo_coord[-1, :]
        homo_coord[[0, 1]] = homo_coord[[1, 0]] # exchange x and y

        coord = np.round((homo_coord.T)[:, :-1])


        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        upper_bound = np.zeros_like(coord)
        upper_bound[:, 0], upper_bound[:, 1] = h_src, w_src

        if interpolation == 'bilinear':
            # sloppily, ignore the pixels on the boundary :-/
            mask = np.logical_and(coord >= 1, coord < (upper_bound - 1))
        else:
            mask = np.logical_and(coord >= 0, coord < upper_bound)
        mask = np.logical_and(mask[:, 0], mask[:, 1])


        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        filtered_coord = coord[mask]
        floor_coord = (np.floor(filtered_coord)).astype(np.int32)
        delta = filtered_coord - floor_coord
        delta = np.expand_dims(delta, axis=2)


        # TODO: 6. assign to destination image with proper masking
        output = (np.copy(dst[ymin:ymax, xmin:xmax, :])).reshape(-1, ch)
        if interpolation == 'bilinear':
            output[mask] = src[floor_coord[:, 0], floor_coord[:, 1], :] * (1 - delta[:, 0]) * (1 - delta[:, 1]) + \
                            src[floor_coord[:, 0], floor_coord[:, 1] + 1, :] * delta[:, 0] * (1 - delta[:, 1]) + \
                                src[floor_coord[:, 0] + 1, floor_coord[:, 1], :] * (1 - delta[:, 0]) * delta[:, 1] + \
                                    src[floor_coord[:, 0] + 1, floor_coord[:, 1] + 1, :] * delta[:, 0] * delta[:, 1]
        else:
            filtered_coord = filtered_coord.astype(np.int32)
            output[mask] = src[filtered_coord[:, 0], filtered_coord[:, 1], :]

        dst[ymin:ymax, xmin:xmax, :] = output.reshape(ymax-ymin, xmax-xmin, ch)


    elif direction == 'f':
        '''
        forward warping
            :var homo_coord:        (3, N).  N equals to (ymax-ymin) * (xmax-xmin)
            :var warped_homo_coord: (3, N)
            :var warped_coord:      (N, 2)
            :var mask:              (N,)
            :var filtered_coord     (M, 2). M is the number of elements passing the mask.
        '''
        homo_coord = np.vstack((x.flatten(), y.flatten(), np.ones((1, x.size))))

        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        warped_homo_coord = np.dot(H, homo_coord)
        warped_homo_coord /= warped_homo_coord[-1, :]
        warped_homo_coord[[0, 1]] = warped_homo_coord[[1, 0]]   # exchange x and y

        warped_coord = np.round((warped_homo_coord.T)[:, :-1]).astype(np.int32)


        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        upper_bound = np.zeros_like(warped_coord)
        upper_bound[:, 0], upper_bound[:, 1] = h_dst, w_dst
        mask = np.logical_and(warped_coord >= 0, warped_coord < upper_bound)
        mask = np.logical_and(mask[:, 0], mask[:, 1])


        # TODO: 5.filter the valid coordinates using previous obtained mask
        mask = mask.reshape(ymax-ymin, xmax-xmin)
        warped_coord = warped_coord.reshape(ymax-ymin, xmax-xmin, -1)
        filtered_coord = warped_coord[mask]


        # TODO: 6. assign to destination image using advanced array indicing
        # for i in range(ymin, ymax):
        #     for j in range(xmin, xmax):
        #         if mask[i, j]:
        #             dst[warped_coord[i, j, 0], warped_coord[i, j, 1], :] = src[i, j, :]
        dst[filtered_coord[:, 0], filtered_coord[:, 1], :] = src[mask]

    return dst
