import numpy as np

import unittest



def PCA(a, K = None, epslon = 0.01):
    a = np.asarray(a)
    A = a.dot(a.T)
    u, sigma, vt = np.linalg.svd(A)
    if K == None:
        err = 0x3f3f3f3f
        K = 0
        sums = np.sum(sigma)
        while err > epslon and K < min(a.shape):
            K += 1
            numerator = 0.0
            for i in range(K):
                numerator += sigma[i]
            err = numerator / sums
    
    P = np.asarray(u[:K, :])

    return P.dot(a)

class PCAUnitTest(unittest.TestCase):

    def test_pca(self):
        data = np.asarray([[-1, -1, 0, 2, 0], [-2, 0, 0, 1, 1]])
        result = np.asarray([[2.12132034, 0.70710678, 0, -2.12132034, -0.70710678]])
        r = PCA(data, K = 1)
        self.assertEqual(r.shape, result.shape)
        self.assertEqual(len(r.shape), 2)
        for i in range(r.shape[0]):
            self.assertLess(abs(result[0, i] - r[0, i]), 1e-6)


if __name__ == "__main__":
    unittest.main()