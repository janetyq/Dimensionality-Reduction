import numpy as np
from utils.data_io import load_csv_data
from utils.plotting import plot_result

def pca_svd(X, num_dim):
    centered_X = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(centered_X, full_matrices=False)
    return centered_X @ Vt.T[:, :num_dim]

def pca_eig(X, num_dim):
    centered_X = X - np.mean(X, axis=0)
    cov = centered_X.T @ centered_X
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    return centered_X @ eigenvectors[:, :num_dim]

def pca(X, num_dim):
    '''
    Principal component analysis
    X: (N, D) matrix
    num_dim: number of dimensions to reduce to
    returns X projected onto the first num_dim principal components (N, num_dim)
    '''
    return np.real(pca_eig(X, num_dim))

if __name__ == '__main__':
    N = 1500
    file_path = 'data/mnist_test.csv'
    label, X = load_csv_data(file_path, N)  # X: (N, 784) orig data

    # PCA reduces to 2 features
    Y = pca(X, 2)                           # Y: (N, 2) low-dim data
    plot_result(Y, label, title='PCA on MNIST digits')

    print('pca.py done')