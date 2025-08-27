# Copyright (c) 2025, Mohammadjavad Vakili

r"""Portfolio optimization module.

This module provides:

- `Portfolio`: A class representing a portfolio of assets and liabilities


As a convention, we use the following notation:

- $\mathbf{R}$: The vector of expected returns of assets and liabilities.
This is the `expected_returns` parameter needed to instantiate the `Portfolio` class.

$$
\mathbf{R} = \begin{bmatrix} r_1 \\ r_2 \\ \vdots \\ r_n \\ r_L \end{bmatrix},
$$

where $r_1, r_2, \ldots, r_n$ are the expected returns of the assets and $r_L$ is the expected return of the liabilities.
The last element of $\mathbf{R}$ always corresponds to the liabilities.


- $\Sigma$: The total covariance matrix consisting of the covariance matrix of asset returns, the
covariance between asset returns and liability returns, and the variance of liability returns.
This is the `covariance_matrix` parameter needed to instantiate the `Portfolio` class.

$$
\mathbf{\Sigma} = \begin{bmatrix}
 \Sigma_{A} , \Sigma_{AL} \\
 \Sigma_{AL} , \sigma^{2}_{L}
\end{bmatrix},
$$

where \( \Sigma_{A} \) is a covariance matrix of the asset returns,
\( \Sigma_{AL} \) is the covariance between the assets and liabilities,
and \( \sigma^{2}_{L} \) is the variance of the liabilities.


- $\mathbf{W}$: The vector of weights of the assets and liabilities in the portfolio it is the output of the optimizers.
The last element corresponds to the liability and it is always set to -1:

$$
\mathbf{W} = \begin{bmatrix}
 w_1 \\
 w_2 \\
 \vdots \\
 w_n \\
 -1
\end{bmatrix},
$$
where \( w_1, w_2, \ldots, w_n \) are the weights of the assets and -1 is the weight of the liabilities.
"""
from typing import Self

import numpy as np
from pydantic import BaseModel, model_validator


class Portfolio(BaseModel):
    r"""Portfolio class.

    This class can be used to instantiate a portfolio object with information about its assets and liabilities.
    Let's assume that we have a portfolio of $n$. The following parameters are required to define the portfolio:

    - `names`: A list with length of $n + 1$ consisting of asset names in the portfolio and the liability.
    Example: `["Asset 1", "Asset 2", ... , "Asset n", "Liability"]`

    - `expected_returns`: An array of length $n + 1$ consisting of expected returns for the assets.

    $$
    \mathbf{R} = \begin{bmatrix} r_1 \\ r_2 \\ \vdots \\ r_n \\ r_L \end{bmatrix},
    $$

    where $r_1, r_2, \ldots, r_n$ are the expected returns of the assets and $r_L$ is the 
    expected return of the liabilities.
    Example: `np.array([0.1, 0.2, ... , 0.1, 0.05])`, where the last element is the expected return of the liabilities.

    - `covariance_matrix`: The total covariance matrix of the asset and liability returns.
    $$
    \mathbf{\Sigma} = \begin{bmatrix} \Sigma & \Sigma_{AL} \\ \Sigma_{AL} & \sigma_L^2 \end{bmatrix},
    $$
    where $\Sigma$ is the $n$ by $n$ covariance matrix of the asset returns, $\Sigma_{AL}$ is an $n$-dimensional vector
    representing the covariance between the assets and liabilities, and $\sigma_L^2$ is the variance of the liabilities.

    Example: `np.array([[0.1, 0.02, ...], [0.02, 0.1, ...], [...], [0.01, 0.005, ...]])`,
    where the last row and column correspond to the liabilities.

    Attributes
    ----------
    names : list[str]
        List of asset names in the portfolio.
    covariance_matrix : np.ndarray
        Covariance matrix of the asset returns.
    expected_returns : np.ndarray
        Expected returns of the assets in the portfolio.

    Properties
    ----------
    n_assets : int
        The number of assets in the portfolio, derived from the length of `names`.

    Methods
    -------
    validate_covariance_matrix() -> Self
        Validates the covariance matrix for shape, symmetry, and positive semi-definiteness.

    validate_expected_returns() -> Self
        Validates the expected returns array for shape consistency with the number of assets.

    validate_weights(weights: np.ndarray) -> None
        Validates the weights of the assets in the portfolio.

    surplus_return(weights: np.ndarray) -> float
        Calculates the surplus return of the portfolio given the asset weights.

    surplus_variance(weights: np.ndarray) -> float
        Calculates the surplus variance of the portfolio given the asset weights.

    portfolio_return(weights: np.ndarray) -> float
        Calculates the return of the portfolio given the asset weights.

    portfolio_variance(weights: np.ndarray) -> float
        Calculates the variance of the portfolio given the weights.

    Raises
    ------
    ValueError
        If the covariance matrix is not square, not symmetric, or not positive semi-definite,
        or if the expected returns array does not match the number of assets.

    """

    names: list[str]
    covariance_matrix: np.ndarray
    expected_returns: np.ndarray

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @property
    def n_assets(self) -> int:
        """Get the number of assets in the portfolio.

        Returns
        -------
        int
            The number of assets in the portfolio.

        """
        return len(self.names) - 1

    @model_validator(mode="after")
    def validate_covariance_matrix(self) -> Self:
        """Validate the covariance matrix of the portfolio.

        Returns
        -------
        Self
            The validated PortfolioModel instance.

        Raises
        ------
        ValueError
            If the covariance matrix is not square, not symmetric, not positive semi-definite, or when
            it does not have the right dimensions.

        """
        if self.covariance_matrix.shape != (self.n_assets + 1, self.n_assets + 1):
            msg = "Covariance matrix must be square with dimensions equal to the number of assets + 1."
            raise ValueError(msg)
        if self.covariance_matrix.ndim != 2:  # noqa: PLR2004
            msg = "Covariance matrix must be a 2D array."
            raise ValueError(msg)
        if not np.allclose(self.covariance_matrix, self.covariance_matrix.T):
            msg = "Covariance matrix must be symmetric."
            raise ValueError(msg)
        if not np.all(np.linalg.eigvals(self.covariance_matrix) >= 0):
            msg = "Covariance matrix must be positive semi-definite."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_expected_returns(self) -> Self:
        """Validate the expected returns of the portfolio.

        Returns
        -------
        Self
            The validated PortfolioModel instance.

        Raises
        ------
        ValueError
            If the expected returns array does not match the number of assets.

        """
        if self.expected_returns.shape != (self.n_assets + 1,):
            msg = "Expected returns must be a 1D array with length equal to the number of assets + 1."
            raise ValueError(msg)
        return self

    def validate_weights(self, weights: np.ndarray) -> None:
        """Validate the weights of the portfolio.

        Parameters
        ----------
        weights : np.ndarray
            The weights of the assets in the portfolio.

        Raises
        ------
        ValueError
            If the weights do not match the number of assets, do not sum to zero,
            or if the last weight is not -1 (for liabilities).

        """
        if len(weights) != self.n_assets + 1:
            msg = "Weights must match the number of assets."
            raise ValueError(msg)
        if not np.isclose(np.sum(weights), 0):
            msg = "Weights must sum to zero."
            raise ValueError(msg)
        if weights[-1] != -1:
            msg = "Last weight must be -1 (for liabilities)."
            raise ValueError(msg)

    def surplus_return(self, weights: np.ndarray) -> float:
        r"""Calculate the surplus return of the portfolio given the asset weights.

        The surplus return is defined as the return of the portfolio - the expected return of the liabilities.
        $$
        R_s = R_p - R_L = \sum_{i=1}^{n} w_i R_i - R_L = \mathbf{W}^{T} \mathbf{R},
        $$
        where $R_i$ is the expected return of asset $i$, $R_L$ is the expected return of the liabilities,
        $R = \begin{bmatrix} R_1 \\ R_2 \\ \vdots \\ R_n \\ R_L \end{bmatrix}$ is `self.expected_returns`
        containing the expected returns of the assets and liabilities,
        and $\mathbf{W} = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \\ -1 \end{bmatrix}$ is the vector of weights.

        Parameters
        ----------
        weights : np.ndarray
            The weights of the assets in the portfolio.

        Returns
        -------
        float
            The surplus return of the portfolio over the expected liabilities return.

        """
        self.validate_weights(weights)
        return float(weights.T @ self.expected_returns)

    def surplus_variance(self, weights: np.ndarray) -> float:
        r"""Calculate the surplus variance of the portfolio given the asset weights.

        The surplus variance is defined as the variance of the surplus return.

        $$
        \sigma^{2}_{s} = \mathbf{W}^{T} \mathbf{\Sigma} \mathbf{W} = \sum_{i=1}^{n_{assets}} \sum_{j=1}^{n_{assets}} w_{i} w_{j} \big(\Sigma_{A}\big)_{ij} - 2 \sum_{i=1}^{n_{assets}} w_{i} \big(\Sigma_{AL}\big)_{i} + \sigma^{2}_{L}
        $$
        where \( \mathbf{\Sigma} \) is the covariance matrix of the asset returns and liabilities, 
        \( \mathbf{\Sigma}_{A} \) is the covariance matrix of the asset returns, \( \mathbf{\Sigma}_{AL} \) is the covariance vector between the assets and liabilities, and \( \sigma^{2}_{L} \) is the variance of the liabilities.

        Parameters
        ----------
        weights : np.ndarray
            The weights of the assets in the portfolio.

        Returns
        -------
        float
            The surplus variance of the portfolio.

        """  # noqa: E501
        self.validate_weights(weights)
        return float(weights.T @ self.covariance_matrix @ weights)

    def portfolio_return(self, weights: np.ndarray) -> float:
        r"""Calculate the return of the portfolio given the asset weights.

        The return of the portfolio is defined as the weighted sum of the returns of the assets.

        $$
        R_p = \sum_{i=1}^{n_{}} w_i R_i = \mathbf{W}_{assets}^{T} \mathbf{R}_{assets},
        $$

        Parameters
        ----------
        weights : np.ndarray
            The weights of the assets in the portfolio.

        Returns
        -------
        float
            The return of the portfolio.

        """
        self.validate_weights(weights)
        return float(weights[:-1].T @ self.expected_returns[:-1])

    def portfolio_variance(self, weights: np.ndarray) -> float:
        r"""Calculate the variance of the portfolio given the weights.

        The variance of the portfolio is defined as.

        $$
        \sigma^{2}_{p} = \mathbf{W}_{assets}^{T} \mathbf{\Sigma}_{A} \mathbf{W}_{assets}
        $$

        Parameters
        ----------
        weights : np.ndarray
            The weights of the assets in the portfolio.

        Returns
        -------
        float
            The variance of the portfolio.

        """
        self.validate_weights(weights)
        return float(weights[:-1].T @ self.covariance_matrix[:-1, :-1] @ weights[:-1])
