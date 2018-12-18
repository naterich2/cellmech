import numpy as np
import numpy.random as npr

A = npr.random((10, 10))
B = npr.random((10, 10, 2))

inds = np.where(A > 0.5)

inds += (np.full(len(inds[0]), 0),)

print B[inds]
