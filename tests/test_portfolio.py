from penfolioop.portfolio import Portfolio

import numpy as np

import pytest 


@pytest.fixture
def expected_returns():
    return np.array([0.05, 0.07, 0.06, 0.04])

@pytest.fixture
def covariance_matrix():
    
    return np.array([
        [0.0004, 0.0002, 0.0001, 0.0003],
        [0.0002, 0.0005, 0.0003, 0.0002],
        [0.0001, 0.0003, 0.0006, 0.0001],
        [0.0003, 0.0002, 0.0001, 0.0004]
    ])

@pytest.fixture
def names():
    return ["Asset A", "Asset B", "Asset C", "Liability"]

@pytest.fixture
def portfolio(expected_returns, covariance_matrix, names):
    return Portfolio(
        names=names,
        covariance_matrix=covariance_matrix,
        expected_returns=expected_returns
    )


def test_incorrect_covariance():
    with pytest.raises(ValueError):
        Portfolio(
            names=["Asset A", "Asset B"],
            covariance_matrix=np.array([[0.0004, 0.0002], [-0.0002, 0.0005]]),
            expected_returns=np.array([0.05, 0.07])
        )

    with pytest.raises(ValueError):
        Portfolio(
            names=["Asset A", "Asset B"],
            covariance_matrix=np.array([[0.0004], [0.0002]]),
            expected_returns=np.array([0.05, 0.07])
        )
    
    with pytest.raises(ValueError):
        Portfolio(
            names=["Asset A", "Asset B"],
            covariance_matrix=np.array([[0.0004, 0.0], [0.0, -0.0005]]),
            expected_returns=np.array([0.05, 0.07])
        )


def test_incorrect_expected_returns():
    with pytest.raises(ValueError):
        Portfolio(
            names=["Asset A", "Asset B"],
            covariance_matrix=np.array([[0.0001, 0.0], [0.0, 0.0005]]),
            expected_returns=np.array([0.05])
        )


def test_portfolio_weights(portfolio):
    weights = np.array([0.4, 0.3, 0.2, -1.0])
    with pytest.raises(ValueError):
        portfolio.validate_weights(weights)

    weights = np.array([0.4, 0.3, 0.3, 0.0])
    with pytest.raises(ValueError):
        portfolio.validate_weights(weights)

    weights = np.array([0.4, 0.3, 0.3, -1.0])
    assert portfolio.validate_weights(weights) is None

    weights = np.array([0.4, 0.6, -1.0])
    with pytest.raises(ValueError):
        portfolio.validate_weights(weights)


def test_portfolio_calculations(portfolio):
    weights = np.array([0.4, 0.3, 0.3, -1.0])
    
    surplus_return = portfolio.surplus_return(weights)
    assert isinstance(surplus_return, float)
    assert surplus_return == weights.T @ portfolio.expected_returns
    
    surplus_variance = portfolio.surplus_variance(weights)
    assert isinstance(surplus_variance, float)
    assert surplus_variance == float(weights.T @ portfolio.covariance_matrix @ weights)
    
    portfolio_return = portfolio.portfolio_return(weights)
    assert isinstance(portfolio_return, float)
    assert portfolio_return == surplus_return - weights[-1] * portfolio.expected_returns[-1]


    portfolio_variance = portfolio.portfolio_variance(weights)
    assert isinstance(portfolio_variance, float)
    assert portfolio_variance == float(weights[:-1].T @ portfolio.covariance_matrix[:-1, :-1] @ weights[:-1])

    # Check if the calculations are consistent
    assert surplus_return == pytest.approx(portfolio_return + float(weights[-1] * portfolio.expected_returns[-1]))
    assert surplus_variance == pytest.approx(portfolio_variance + float(weights[-1]**2 * portfolio.covariance_matrix[-1, -1] + 2 * weights[-1] * portfolio.covariance_matrix[:-1, -1].T @ weights[:-1]))
