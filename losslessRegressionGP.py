from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

lambda_f = 1.0
l_square = 10.0


def kernel(x, x_prime):
    """ GP Squared exponential kernel """
    x_sqr = np.sum(x**2, 1).reshape(-1, 1)
    #print(f"x_sqr shape: {x_sqr.shape}, {x_sqr}")
    x_prime_sqr = np.sum(x_prime**2, 1)
    #print(f"x_prime_sqr shape: {x_prime_sqr.shape}, {x_prime_sqr}")
    x_x_prime_dot = 2*np.dot(x, x_prime.T)
    #print(f"x_x_prime_dot shape: {x_x_prime_dot.shape}, {x_x_prime_dot}")

    sqdist = x_sqr + x_prime_sqr - x_x_prime_dot
    #print(f"sqdist_shape: {sqdist.shape}, {sqdist}")
    sigma_ = np.exp((-0.5/l_square) * sqdist)
    #print(f"sigma_ shape: {sigma_.shape}, {sigma_}")
    return sigma_


def main():
    n = int(input("Enter the number of points for simulation: "))
    x_test = np.linspace(-5, 5, n).reshape(-1, 1)
    #print(f"test set shape: {x_test.shape}, {x_test}")

    # Kernel at test points
    sigma_ = kernel(x_test, x_test)

    # Cholesky decom matrix
    L = validate_positive_definitive(sigma_)
    #print(L)
    #L = np.linalg.cholesky(sigma_ + 1e-6 * np.eye(n))

    # Draw N(0,1) samples from the prior at our test points. 10 samples for each data point
    f_prior = (lambda_f **2) * np.dot(L, np.random.normal(size=(n, 10)))

    # Plot
    plt.plot(x_test, f_prior)
    plt.show()


def to_positive_definitive(M):
    M = np.matrix(M)
    M = (M + M.T) * 0.5
    k = 1
    I = np.eye(M.shape[0])
    w, v = np.linalg.eig(M)
    min_eig = v.min()
    M += (-min_eig * k * k + np.spacing(min_eig)) * I
    return M


def validate_positive_definitive(M):
    try:
        np.linalg.cholesky(M + 1e-6 * np.eye(M.shape[0]))
    except np.linalg.LinAlgError:
        print("Not positive definite matrix, so converting it")
        M = to_positive_definitive(M)

    # Print the eigenvalues of the Matrix
    #print(np.linalg.eigvalsh(M))
    return M


if __name__ == '__main__':
    main()
