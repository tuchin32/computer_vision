import numpy as np
import cv2.ximgproc as xip

def getNeighbors(I):
    # Pad the image
    h, w, ch = I.shape
    padded_I = np.pad(I, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0) # (h+2, w+2, ch)

    # Store the local binary pattern of each pixel in the neighborhood
    # Window size = 3 x 3 (8 neighbors)
    neighbors_I = np.zeros((h, w, 8, ch), dtype=np.bool)
    neighbors_I[:, :, 0] = padded_I[1:-1, :-2] < padded_I[1:-1, 1:-1] # left
    neighbors_I[:, :, 1] = padded_I[2:, :-2] < padded_I[1:-1, 1:-1] # left-down
    neighbors_I[:, :, 2] = padded_I[2:, 1:-1] < padded_I[1:-1, 1:-1] # down
    neighbors_I[:, :, 3] = padded_I[2:, 2:] < padded_I[1:-1, 1:-1] # right-down
    neighbors_I[:, :, 4] = padded_I[1:-1, 2:] < padded_I[1:-1, 1:-1] # right
    neighbors_I[:, :, 5] = padded_I[:-2, 2:] < padded_I[1:-1, 1:-1] # right-up
    neighbors_I[:, :, 6] = padded_I[:-2, 1:-1] < padded_I[1:-1, 1:-1] # up
    neighbors_I[:, :, 7] = padded_I[:-2, :-2] < padded_I[1:-1, 1:-1] # left-up

    return neighbors_I


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # Gray image: (h, w)
    Il_gray = (np.average(Il, axis=2, weights=[0.114, 0.587, 0.299])).astype(np.float32)
    Ir_gray = (np.average(Ir, axis=2, weights=[0.114, 0.587, 0.299])).astype(np.float32)

    # Neigbirhood binary pattern: (h, w, 8, ch)
    n_Il = getNeighbors(Il)
    n_Ir = getNeighbors(Ir)


    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    cost_l = np.zeros((h, w, max_disp + 1), dtype=np.float32)
    cost_r = np.zeros((h, w, max_disp + 1), dtype=np.float32)
    for d in range(max_disp + 1):
        if d == 0:
            cost_l[:, :, d] = np.sum(np.logical_xor(n_Il, n_Ir).astype(np.uint8), axis=(2, 3))
            cost_r[:, :, d] = np.sum(np.logical_xor(n_Ir, n_Il).astype(np.uint8), axis=(2, 3))
        else:
            cost_l[:, d:, d] = np.sum(np.logical_xor(n_Il[:, d:], n_Ir[:, :-d]).astype(np.uint8), axis=(2, 3))
            cost_l[:, :d, d] = cost_l[:, d:d+1, d]
            cost_r[:, :-d, d] = np.sum(np.logical_xor(n_Ir[:, :-d], n_Il[:, d:]).astype(np.uint8), axis=(2, 3))
            cost_r[:, -d:, d] = cost_r[:, -d-1:-d, d]


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    sigma_color, sigma_space = 4, 11
    jbf_cost_l = np.zeros_like(cost_l)
    jbf_cost_r = np.zeros_like(cost_r)
    for d in range(max_disp + 1):
        jbf_cost_l[:, :, d] = xip.jointBilateralFilter(Il_gray, cost_l[:, :, d], -1, sigma_color, sigma_space)
        jbf_cost_r[:, :, d] = xip.jointBilateralFilter(Ir_gray, cost_r[:, :, d], -1, sigma_color, sigma_space)
    

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    labels_l = np.argmin(jbf_cost_l, axis=2)
    labels_r = np.argmin(jbf_cost_r, axis=2)
    
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    
    # Left-right consistency check
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    mask = (labels_l[y, x] == labels_r[y, x - labels_l[y, x]]).astype(np.uint8)

    valid_labels_l = np.where(mask == 1, labels_l, -1)
    valid_labels_l = np.pad(valid_labels_l, ((0, 0), (1, 1)), 'constant', constant_values=max_disp - 1)
    refine_ll = np.where(mask == 1, labels_l, 0)
    refine_lr = np.where(mask == 1, labels_l, 0)

    # Hole filling
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0:
                # Fill refine_ll
                j_idx = np.where(valid_labels_l[i, :j+1] != -1)[0][-1]
                refine_ll[i, j] = valid_labels_l[i, j_idx]

                # Fill refine_lr
                j_dx = np.where(valid_labels_l[i, j+2:] != -1)[0][0] + j + 2
                refine_lr[i, j] = valid_labels_l[i, j_dx]

    labels = np.min((refine_ll, refine_lr), axis=0)

    # Weighted median filtering
    window_radius = 3
    labels = xip.weightedMedianFilter(Il_gray.astype(np.uint8), labels.astype(np.uint8), window_radius)


    return labels.astype(np.uint8)