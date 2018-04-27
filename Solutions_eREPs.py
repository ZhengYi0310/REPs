import numpy as np

mean = np.array([0, 1])
cov = np.array([[1, 0], [0, 100]])
print cov[0, :].shape
print np.linalg.inv(cov)
print mean * cov

# import matplotlib.pyplot as plt
# a = np.random.RandomState(1234)
# b = a.multivariate_normal(mean, cov, 1)
# print np.finfo(float).eps

# m, n = 100, 100
# lims = (-3, 3) # support of the PDF
# xx, yy = np.meshgrid(np.linspace(-3, 3, m), np.linspace(-3, 3, n))
# points = np.stack((xx, yy), axis=-1)
# print points.shape
