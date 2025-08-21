# PenFolioOp
Portfolio Optimizations for *Pension Funds*

## Supported optimizers

- Surplus Return Maximization
- Surplus Variance Minimization
- Surplus Sharpe Ratio Maximization
- Mean-Variance Optimization

Surplus return is defined as the return of the portfolio minus the return of the liabilities, while surplus variance is the variance of the portfolio returns in excess of the liabilities. 

## Supported constraints

Asset weight constraints can be applied to ensure that the portfolio adheres to specific investment guidelines. 

Usage of the library is straightforward. You can create a portfolio object, define your assets, and then use the optimizers to find the optimal asset weights based on your constraints and objectives.



## Example


The first step is to create a `Portfolio` object with your assets, their expected returns, and covariances. The last item in the list of assets should be the liability, which is treated differently from the other assets in the optimization process. The optimizaters always set the liability weight to -1 and require the other asset weights to be between 0 and 1 and sum to 1.

The user can then define additional constraints on the asset weights, such as requiring a minimum or maximum weight for certain assets or limiting the weight of one asset to be less than another.

For a comprehensive description of the constraints, refer to the API documentation.


```python
from penfolioop import Portfolio, SurplusReturnMaximization

names = ['Asset A', 'Asset B', 'Asset C', 'Liability']
returns = [0.05, 0.07, 0.06, 0.04]
covariances = [[0.0001, 0.00005, 0.00002, 0.00003],
               [0.00005, 0.0002, 0.00001, 0.00004],
               [0.00002, 0.00001, 0.00015, 0.00002],
               [0.00003, 0.00004, 0.00002, 0.0001]]

portfolio = Portfolio(names, returns, covariances)

constraints = [
    {
        'left_indices': ['Asset A', 'Asset B'],
        'operator': '>=',
        'right_value': 0.5
    }
    {
        'left_indices': ['Asset C'],
        'operator': '<=',
        'right_index': ['Asset B']
    }
]

weights = SurplusReturnMaximization(portfolio=portfolio, asset_constraints=constraints)
``` 

