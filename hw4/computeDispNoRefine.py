import numpy as np
import cv2
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

    ##############################################

    Il_gray = (np.mean(Il, axis=2)).astype(np.float32)

    # neighbor_I: (h, w, 8, ch)
    n_Il = getNeighbors(Il)
    n_Ir = getNeighbors(Ir)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    cost_l = np.zeros((h, w, max_disp), dtype=np.float32)
    for d in range(max_disp):
        if d == 0:
            cost_l[:, :, d] = np.sum(np.logical_xor(n_Il, n_Ir).astype(np.uint8), axis=(2, 3))
        else:
            cost_l[:, d:, d] = np.sum(np.logical_xor(n_Il[:, d:], n_Ir[:, :-d]).astype(np.uint8), axis=(2, 3))
            cost_l[:, :d, d] = cost_l[:, d:d+1, d]

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    jbf_cost_l = np.zeros_like(cost_l)
    for d in range(max_disp):
        jbf_cost_l[:, :, d] = xip.jointBilateralFilter(Il, cost_l[:, :, d], -1, 50, 24)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    labels = np.argmin(jbf_cost_l, axis=2)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    pass

    return labels.astype(np.uint8)