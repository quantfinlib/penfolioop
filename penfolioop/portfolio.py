# Copyright (c) 2025, Mohammadjavad Vakili

"""Portfolio optimization module.

This module provides:
- Portfolio: A dataclass representing a portfolio of assets and liabilities
  with methods for calculating returns, variance, and surplus metrics.
"""
from typing import Self

import numpy as np
from pydantic import BaseModel, model_validator


class Portfolio(BaseModel):
    """Base model for Portfolio, used for validation.
    
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
        return len(self.names)

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
            If the covariance matrix is not square, not symmetric, or not positive semi-definite.

        """
        if self.covariance_matrix.shape != (self.n_assets, self.n_assets):
            msg = "Covariance matrix must be square with dimensions equal to the number of assets."
            raise ValueError(msg)
        if self.covariance_matrix.ndim != 2:
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
        if self.expected_returns.shape != (self.n_assets,):
            msg = "Expected returns must be a 1D array with length equal to the number of assets."
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
        if len(weights) != self.n_assets:
            msg = "Weights must match the number of assets."
            raise ValueError(msg)
        if not np.isclose(np.sum(weights), 0):
            msg = "Weights must sum to zero."
            raise ValueError(msg)
        if weights[-1] != -1:
            msg = "Last weight must be -1 (for liabilities)."
            raise ValueError(msg)

    def surplus_return(self, weights: np.ndarray) -> float:
        """Calculate the surplus return of the portfolio given the asset weights.

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
        """Calculate the surplus variance of the portfolio given the asset weights.

        Parameters
        ----------
        weights : np.ndarray
            The weights of the assets in the portfolio.

        Returns
        -------
        float
            The surplus variance of the portfolio.

        """
        self.validate_weights(weights)
        return float(weights.T @ self.covariance_matrix @ weights)

    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate the return of the portfolio given the asset weights.

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
        """Calculate the variance of the portfolio given the weights.

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

