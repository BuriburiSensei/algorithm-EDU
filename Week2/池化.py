import numpy as np


def pooling_process(imputs,pool_size,stride,mode='max'):
    h,w = imputs.shape

    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    output = np.zeros((out_h,out_w))

    for r in range(out_h):
        for c in range(out_w):
            r_start = r * stride
            r_end = r_start + pool_size
            c_start = c * stride
            c_end = c_start + pool_size
            window = imputs[r_start:r_end, c_start:c_end]
            if mode == 'max':
                output[r,c] = np.nanmax(window)
            elif mode == 'avg':
                output[r,c] = np.nanmean(window)

    return output