import numpy as np





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

if __name__ == "__main__":
    data = np.asarray([[-1, -1, 0, 2, 0], [-2, 0, 0, 1, 1]])
    print PCA(data, K = 1)