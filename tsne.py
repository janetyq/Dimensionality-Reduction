'''
Implementation of t-SNE (t-distributed Stochastic Neighbor Embedding)
'''

import numpy as np
from pca import pca
from utils.data_io import load_csv_data
from utils.plotting import plot_result

def calculate_sqnorm(X):
    '''
    Calculates pairwise squared norm of each row of X, efficiently
    '''
    XXT = X @ X.T
    diag = np.diag(XXT)[:, np.newaxis]
    sqnorm = diag @ np.ones((1, N)) - 2 * XXT + np.ones((N, 1)) @ diag.T 
    return sqnorm

def calculate_diff_and_sqnorm(X):
    X_diff = X[:, None, :] - X[None, :, :]
    X_sqnorm = calculate_sqnorm(X)
    return X_diff, X_sqnorm

def calculate_entropy(sigmas, sqnorm):
    '''
    Calculates Shannon entropy of prob dist for each point in X
    '''
    similarity = np.exp(-sqnorm / (2 * (sigmas[:, np.newaxis])**2))
    P = (similarity / (np.sum(similarity, axis=1)[:, None] - 1))
    entropy = -np.sum(P * np.log2(P), axis=1) # entropy of each row
    return entropy

def optimize_sigmas(sqnorm, target_perp, bounds=None, max_iter=50, tol=1e-4):
    '''
    Find sigmas that give perplexity = target_perp
    sigmas: list of stdev of gaussian prob dist around each point
    perplexity: effective number of neighbors around each point
    '''

    if bounds is None:
        bounds = (0, len(X[0]))

    N = len(X)
    sigmas_low = np.full(N, bounds[0])
    sigmas_high = np.full(N, bounds[1])
    sigmas = (sigmas_low + sigmas_high) / 2
    target_entropy = np.log2(target_perp) # 2**entropy = perplexity
    for _ in range(max_iter):
        entropy = calculate_entropy(sigmas, sqnorm)
        update_mask = (entropy > target_entropy)
        sigmas_low = update_mask * sigmas_low + ~update_mask * sigmas
        sigmas_high = update_mask * sigmas + ~update_mask * sigmas_high
        sigmas = update_mask * (sigmas + sigmas_low) / 2 + ~update_mask * (sigmas + sigmas_high) / 2
        
        if np.all(np.abs(entropy - target_entropy) < tol):
            break

    return sigmas

def calculate_symmetric_distribution(sigmas, sqnorm):
    '''
    Calculates symmetric prob dist matrix P for given sigmas
    Guarantees minimum contribution of 1/(2*N) for each point
    '''
    X_similarity = np.exp(-sqnorm / (2 * (sigmas[:, np.newaxis])**2))
    P = (X_similarity / (np.sum(X_similarity, axis=1)[:, None] - 1)) # P[i, j] = probability of j given i
    return (P + P.T) / (2*N)


def calculate_P_matrix(X, target_perp):
    '''
    Calculates probability distribution matrix P on 
    probability of each point being another point's neighbor
    P[i, j] = probability of j given 
    Based on symmetric SNE which uses a gaussian distribution
    '''
    X_diff, X_sqnorm = calculate_diff_and_sqnorm(X)
    sigmas = optimize_sigmas(X_sqnorm, target_perp, bounds=(0, 256*256))
    P = calculate_symmetric_distribution(sigmas, X_sqnorm)
    return P

def optimize_Y(Y, P, eta=100, alpha=0, iter=200):
    '''
    Gradient descent and momentum optimization on Y
    to reduce KL-divergence between P (data prob dist) and Q (low-dim prob dist)

    T-SNE uses a t-distribution for low-dim prob dist Q
    '''
    dY = np.zeros_like(Y)
    for i in range(iter):
        Y_diff, Y_sqnorm = calculate_diff_and_sqnorm(Y)
        Y_similarity = np.power(1 + Y_sqnorm, -1)
        Y_total_similarity = np.sum(Y_similarity) - N

        Q = (Y_similarity / Y_total_similarity)

        ee = 4 if i < 100 else 1 # early exaggeration
        dCdy = np.sum(4 * ((ee*P - Q) * Y_similarity)[:, :, np.newaxis] * Y_diff , axis=1)
        dY = -eta * dCdy + alpha * dY
        Y += dY
    C = np.sum(P * np.log(P / Q))
    return Y, C

def tsne(X, target_perp, eta=100, alpha=0, iter=200, do_pca=None):
    '''
    Performs t-SNE on X
    to get low-dim embedding Y and KL-divergence cost C
    '''
    N = len(X)

    # PCA to reduce dimensionality
    if do_pca is not None:
        X = pca(X, do_pca)  # reduces to (N, do_pca)

    # Calculate probability dist matrix for given perplexity
    P = calculate_P_matrix(X, target_perp)

    # Initialize Y low-dim embedding
    Y = np.random.random((N, 2))

    # Optimize Y
    Y, C = optimize_Y(Y, P, eta, alpha, iter)

    # Return low-dim embedding and cost
    return Y, C


if __name__ == '__main__':
    # Load MNIST handwritten digit data
    N = 1500
    file_path = 'data/mnist_test.csv'
    label, X = load_csv_data(file_path, N)  # X: (N, 784) orig data

    # Perplexity: typical neighborhood size of each point
    target_perp = 20                  

    # Run T-SNE
    Y_result, Y_cost = tsne(X, target_perp, eta=100, alpha=0.5, iter=500, do_pca=100)

    # Plot result
    print(f'Final Cost: {Y_cost:.4f}')
    plot_result(Y_result, label, title='t-SNE on MNIST digits')

    print('tsne.py done')