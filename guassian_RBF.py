import numpy as np
import matplotlib.pyplot as plt


def gaussian_rbf(x, center, width):
    """Calculates the Gaussian RBF for each data point."""
    return np.exp(-(x - center)**2 / (2 * width**2))


if __name__ == '__main__':
    print(f"Plot GAUSSIAN RBF with mean: 0 and std dev: 1")

    # Generate some data
    x = np.linspace(-5, 5, 100)
    y = gaussian_rbf(x, 0, 1)

    # Plot the data
    plt.plot(x, y, 'o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Radial Basis Function')
    plt.savefig('gaussian_rbf.png')
    plt.close()
