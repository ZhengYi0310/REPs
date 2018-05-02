import numpy as np
from sklearn.preprocessing import PolynomialFeatures
mean = np.array([[0, 2, 3]])
poly = PolynomialFeatures(2)
print np.stack([2, 2])
print poly.fit_transform(mean)
cov = np.array([[2, 0], [0, 100]])
v, w = np.linalg.eig(cov)
print np.dot(w, np.dot(np.diag(v), w.T))

np.fill_diagonal(cov, cov.diagonal() + 10)
print cov
print np.triu_indices(3, 0)


# import matplotlib.pyplot as plt
# a = np.random.RandomState(1234)
# b = a.multivariate_normal(mean, cov, 1)
# print np.finfo(float).eps

# m, n = 100, 100
# lims = (-3, 3) # support of the PDF
# xx, yy = np.meshgrid(np.linspace(-3, 3, m), np.linspace(-3, 3, n))
# points = np.stack((xx, yy), axis=-1)
# print points.shape
import os
import sys
import logging
from cStringIO import StringIO


from collections import deque

a = deque(maxlen=2)
a.append(mean)
a.append(cov)
print a
