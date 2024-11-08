import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from scipy.stats import multivariate_normal
import os

class GMM():
    """
    A Gaussian Mixture Model (GMM) class for fitting data using the Expectation-Maximization (EM) algorithm.
    
    Parameters:
    -----------
    M : int
        The number of Gaussian components (clusters).
    X : numpy.ndarray
        The input data (N samples, dim dimensions).
    dim : int, optional
        The dimensionality of the input data (default is 2).
    
    Attributes:
    -----------
    mu : numpy.ndarray
        The means of the Gaussian components, shape (M, dim).
    tau : numpy.ndarray
        The responsibility matrix, shape (N, M), indicating the probability of each sample belonging to each cluster.
    pi : numpy.ndarray
        The mixing coefficients (weights) for each Gaussian component.
    sigma : numpy.ndarray
        The covariance matrices of the Gaussian components, shape (M, dim, dim).
    """

    def __init__(self, M, X, dim=2):
        """
        Initialize the GMM model with random means, covariances, and responsibilities.
        
        Parameters:
        -----------
        M : int
            Number of Gaussian components.
        X : numpy.ndarray
            Input data (N samples, dim dimensions).
        dim : int, optional
            Dimensionality of the input data (default is 2).
        """
        self.dim = dim
        self.X   = X  # Input data of shape (N, dim)
        self.N   = len(X)  # Number of samples
        self.M   = M  # Number of Gaussian components

        # Initialize means randomly within the bounds of the data
        self.mu  = np.random.uniform(low=self.X.min(axis=0), high=self.X.max(axis=0), size=(self.M, self.dim))

        # Initialize responsibilities randomly using a Dirichlet distribution
        self.tau = np.random.dirichlet(np.ones(self.M), size=self.N)

        # Initialize the mixing coefficients uniformly (equal weights for all clusters)
        self.pi  = np.ones(self.M) / self.M

        # Initialize covariance matrices as identity matrices for each component
        self.sigma = np.array([np.eye(self.dim) for _ in range(self.M)])

        self.log_likelihoods = []

    def _compute_normal_multivariate_probability(self, x, k):
        """
        Compute the multivariate Gaussian probability density function for a given sample.
        
        Parameters:
        -----------
        x : numpy.ndarray
            A sample data point.
        k : int
            The index of the Gaussian component.
        
        Returns:
        --------
        float
            The probability density value for the sample under the k-th Gaussian component.
        """
        # Compute the normalization factor
        normalization_part = 1/np.sqrt((2*np.pi)**self.dim*(np.linalg.det(self.sigma[k])+1e-9))
        
        # Compute the quadratic form (Mahalanobis distance)
        quadratic_part     = (x-self.mu[k]) @ np.linalg.inv(self.sigma[k]) @ (x-self.mu[k])
        
        # Compute the exponential factor
        exponential_part   = np.exp(-0.5 * quadratic_part)
        
        return normalization_part * exponential_part


    def _init_gmm(self):
        # Initialize means randomly within the bounds of the data
        self.mu  = np.random.uniform(low=self.X.min(axis=0), high=self.X.max(axis=0), size=(self.M, self.dim))

        # Initialize responsibilities randomly using a Dirichlet distribution
        self.tau = np.random.dirichlet(np.ones(self.M), size=self.N)

        # Initialize the mixing coefficients uniformly (equal weights for all clusters)
        self.pi  = np.ones(self.M) / self.M

        # Initialize covariance matrices as identity matrices for each component
        self.sigma = np.array([np.eye(self.dim) for _ in range(self.M)])

        self.log_likelihoods = []

    
    def _compute_nk(self):
        """
        Compute the effective number of points assigned to each Gaussian component.
        
        Returns:
        --------
        numpy.ndarray
            A 1D array where each entry represents the sum of responsibilities for a component.
        """
        return self.tau.sum(axis=0)


    def _compute_tau(self):
        """
        E-step: Update the responsibility matrix `tau` based on the current estimates of
        the model parameters (pi, mu, sigma).
        """
        for k in range(self.M):
            # Update responsibilities based on current parameters
            self.tau[:, k] = self.pi[k] * multivariate_normal(mean=self.mu[k], cov=self.sigma[k]).pdf(self.X)
        self.tau /= self.tau.sum(axis=1, keepdims=True)  # Normalize responsibilities


    def _compute_pi(self):
        """
        M-step: Update the mixing coefficients (weights) `pi`.
        """
        self.pi = self.tau.mean(axis=0)

    def _compute_mu(self):
        """
        M-step: Update the means `mu` of the Gaussian components.
        """
        self.mu = (self.tau.T @ self.X) / self._compute_nk()[:, None]
        

    def _compute_sigma(self):
        """
        M-step: Update the covariance matrices `sigma` of the Gaussian components.
        """
        for k in range(self.M):
            diff = self.X - self.mu[k]  # Shape (N, dim)
            self.sigma[k] = (self.tau[:, k][:, np.newaxis] * diff).T @ diff / self._compute_nk()[k]  # Outer product and normalization



    def _compute_log_likelihood(self):
        """
        Compute the log-likelihood of the current model given the data.
        
        Returns:
        --------
        float
            The log-likelihood of the current model.
        """
        sum = 0
        for k in range(self.M):
            # Sum over the probabilities for all Gaussian components
            prob = self.pi[k] * multivariate_normal(mean=self.mu[k], cov=self.sigma[k]).pdf(self.X)
            sum += self.tau[:, k] @ np.log( prob + 1e-9)  # Add small value to avoid log(0)
        return sum
    

    def _M_step(self):
        """
        Perform the M-step of the EM algorithm, which updates the model parameters
        (pi, mu, sigma) based on the current responsibilities.
        """
        self._compute_pi()
        self._compute_mu()
        self._compute_sigma()


    def _E_step(self):
        """
        Perform the E-step of the EM algorithm, which updates the responsibilities `tau`
        based on the current model parameters.
        """
        self._compute_tau()


    def _em_algorithm(self, max_iter=1000, tol=1e-12, verbose=True):
        """
        Run the Expectation-Maximization (EM) algorithm to fit the GMM to the data.
        
        Parameters:
        -----------
        max_iter : int, optional
            The maximum number of iterations to run (default is 1000).
        tol : float, optional
            The convergence threshold for log-likelihood (default is 1e-12).
        
        Returns:
        --------
        list
            A list of log-likelihood values for each iteration.
        """
        for iteration in range(max_iter):
            old_log_likelihood = self._compute_log_likelihood()
            
            # Perform the M and E steps
            self._M_step()
            self._E_step()
            
            # Compute new log-likelihood
            new_log_likelihood = self._compute_log_likelihood()
            self.log_likelihoods.append(new_log_likelihood)
            
            # Print log-likelihood for tracking
            if verbose:
                print(f"Iteration {iteration + 1}: Log Likelihood = {new_log_likelihood:.6f}")
            
            # Check for convergence
            if np.abs(new_log_likelihood - old_log_likelihood) < tol:
                if verbose:
                    print("Convergence reached.")
                break
                
        return self.log_likelihoods
    

    def fit(self, max_iter=1000, tol=1e-12, verbose=True):
        """
        Fit the GMM to the data using the EM algorithm.
        
        Parameters:
        -----------
        max_iter : int, optional
            The maximum number of iterations to run (default is 1000).
        tol : float, optional
            The convergence threshold for log-likelihood (default is 1e-12).
        """
        self._em_algorithm(max_iter, tol, verbose)


    def bic(self):
        """
        Compute the Bayesian Information Criterion (BIC) for the GMM model.
        
        Returns:
        --------
        float
            The BIC value for the current model.
        """
        # Number of parameters for the model
        num_params = ( self.M - 1 ) + (self.M * self.dim) + (self.M * self.dim * (self.dim + 1) / 2)
        
        # Compute the BIC value
        return - self._compute_log_likelihood() + num_params * np.log(self.N) / 2

    
    def _plot_gaussian_ellipses(self, mu, sigma, ax=None, n_std=2.0, **kwargs):
        """
        Plot Gaussian confidence ellipses for a 2D GMM component.
        
        Parameters:
        -----------
        mu : numpy.ndarray
            Mean of the Gaussian (center of the ellipse), shape (2,).
        sigma : numpy.ndarray
            Covariance matrix of the Gaussian, shape (2, 2).
        ax : matplotlib.axes.Axes, optional
            The plot to draw the ellipses on (default is None, will create a new one).
        n_std : float, optional
            Number of standard deviations to determine the size of the ellipse (default is 2, for 95% confidence).
        kwargs : optional
            Additional keyword arguments for the Ellipse patch (like edgecolor, linestyle).
        
        Returns:
        --------
        matplotlib.patches.Ellipse
            The drawn ellipse object.
        """
        # Ensure that we are plotting in a 2D space
        if sigma.shape != (2, 2):
            raise ValueError("Sigma must be a 2x2 matrix.")
        
        # Eigenvalue decomposition of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(sigma)
        
        # Get the angle of the largest eigenvector
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        
        # Width and height of the ellipse are 2 * sqrt(eigenvalues) * n_std
        width, height = 2 * n_std * np.sqrt(eigvals)
        
        # Create the ellipse
        radius_iter = 8
        for r in np.linspace(0, 1.5, radius_iter):
            ellipse = Ellipse(xy=mu, width=r*width, height=r*height, angle=angle, alpha=(1-(r/1.5)**2), **kwargs)
            
            if ax is None:
                ax = plt.gca()
            
            # Add the ellipse to the plot
            ax.add_patch(ellipse)
        
        return ellipse


    def plot_gmm_ellipses(self, ax=None, colors=None):
        """
        Plot ellipses for all components of a GMM.
        
        Parameters:
        -----------
        gmm : GMM object
            The GMM object containing mu and sigma.
        ax : matplotlib.axes.Axes, optional
            The plot to draw the ellipses on (default is None, will create a new one).
        """
        if ax is None:
            fig, ax = plt.subplots()


         # Plot data points as well
        ax.scatter(self.X[:, 0], self.X[:, 1], s=10, c='black', alpha=0.5)        

        # Plot ellipses for each Gaussian component
        if colors is None:
            colors = np.random.random(size=(self.M, 3)) 
            colors = np.where(colors < 0.5, 1-colors, colors)  # Avoid very dark colors
        for k in range(self.M):
            self._plot_gaussian_ellipses(self.mu[k], self.sigma[k], ax=ax, lw=2, n_std=2, edgecolor=colors[k], facecolor=colors[k])


        


        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title('GMM Gaussian Components')


    def animate_gmm_fit(self, max_iter=1000, tol=1e-12, save=False):
        """
        Animate the GMM fitting process by plotting the data and the Gaussian components at each iteration.
        
        Parameters:
        -----------
        max_iter : int, optional
            The maximum number of iterations to run (default is 1000).
        tol : float, optional
            The convergence threshold for log-likelihood (default is 1e-12).
        save_path : str, optional
            The path to save the animation as an MP4 file (default is None, no saving).
        """
        fig, ax = plt.subplots()
        colors = np.random.random(size=(self.M, 3))
        self._init_gmm()
        
        def update(frame):
            self.fit(max_iter=1, tol=tol, verbose=False)
            ax.clear()
            ax.scatter(self.X[:, 0], self.X[:, 1], s=10, c='black')

            # sort the gaussian component by the means
            idx = np.argsort(self.mu[:, 0])
            self.mu, self.sigma = self.mu[idx], self.sigma[idx]
            
            for k in range(self.M):
                self._plot_gaussian_ellipses(self.mu[k], self.sigma[k], ax=ax, lw=2, n_std=2, alpha=0.01, edgecolor=colors[k], facecolor=colors[k])
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_title(f'GMM Gaussian Components (Iteration {frame})')
            self.plot_gmm_ellipses(ax=ax, colors=colors)

        # Animate the fitting process
        anim = FuncAnimation(fig, update, frames=max_iter, interval=20)

        if save:
            if not os.path.exists("Animation"):
                os.makedirs("Animation")
            anim.save(os.path.join("Animation", "gmm.gif"), writer='pillow', fps=2)

