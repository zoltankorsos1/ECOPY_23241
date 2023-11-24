import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import f


class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side, X).fit()
        self._model = model

    def get_params(self):
        beta_params = self._model.params.rename("Beta coefficients")
        return beta_params

    def get_pvalues(self):
        p_values = self._model.pvalues.rename("P-values for the corresponding coefficients")
        return p_values

    def get_wald_test_result(self, restriction_matrix):
        wald_result = self._model.wald_test(restriction_matrix)
        f_value = round(wald_result.statistic[0, 0], 2)
        p_value = wald_result.pvalue
        result_text = f"F-value: {f_value:.2f}, p-value: {p_value:.3f}"
        return result_text

    def get_model_goodness_values(self):
        ars = self._model.rsquared_adj
        ak = self._model.aic
        by = self._model.bic
        result_text = f"Adjusted R-squared: {ars:.3f}, Akaike IC: {ak:.2e}, Bayes IC: {by:.2e}"
        return result_text


class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.beta = beta

    def get_params(self):
        return pd.Series(self.beta, name='Beta coefficients')

    def get_pvalues(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        n, k = X.shape
        beta = self.beta
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        residuals = y - X @ beta
        residual_variance = (residuals @ residuals) / (n - k)
        standard_errors = np.sqrt(np.diagonal(residual_variance * np.linalg.inv(X.T @ X)))
        t_statistics = beta / standard_errors
        df = n - k
        p_values = [2 * (1 - t.cdf(abs(t_stat), df)) for t_stat in t_statistics]
        p_values = pd.Series(p_values, name="P-values for the corresponding coefficients")
        return p_values

    def get_wald_test_result(self, R):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        beta = self.beta
        residuals = y - X @ beta
        r_matrix = np.array(R)
        r = r_matrix @ beta
        n = len(self.left_hand_side)
        m, k = r_matrix.shape
        sigma_squared = np.sum(residuals ** 2) / (n - k)
        H = r_matrix @ np.linalg.inv(X.T @ X) @ r_matrix.T
        wald = (r.T @ np.linalg.inv(H) @ r) / (m * sigma_squared)
        p_value = 1 - f.cdf(wald, dfn=m, dfd=n - k)
        return f'Wald: {wald:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        n, k = X.shape
        beta = self.beta
        y_pred = X @ beta
        ssr = np.sum((y_pred - np.mean(y)) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        centered_r_squared = ssr / sst
        adjusted_r_squared = 1 - (1 - centered_r_squared) * (n - 1) / (n - k)
        result = f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"
        return result


import pandas as pd
import numpy as np
from scipy.stats import t, f as f_dist

class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        # Initialize the class with response and predictor variables
        self.response_var = left_hand_side.values
        self.predictor_var = right_hand_side.values

    def fit(self):
        # Fit the Generalized Least Squares model
        self.design_matrix = np.column_stack((np.ones(len(self.predictor_var)), self.predictor_var))
        self.observed_values = self.response_var

        # Calculate initial OLS estimates
        beta_initial = np.linalg.inv(self.design_matrix.T @ self.design_matrix) @ self.design_matrix.T @ self.observed_values
        residuals_initial = self.observed_values - self.design_matrix @ beta_initial
        log_resid_initial = np.log(residuals_initial ** 2)

        # Weighted OLS
        beta_weighted = np.linalg.inv(self.design_matrix.T @ self.design_matrix) @ self.design_matrix.T @ log_resid_initial
        weight_diag = 1 / np.sqrt(np.exp(self.design_matrix @ beta_weighted))
        self.weights_inv = np.diag(weight_diag)

        # GLS estimates
        beta_gls = np.linalg.inv(self.design_matrix.T @ self.weights_inv @ self.design_matrix) @ self.design_matrix.T @ self.weights_inv @ self.observed_values
        self.coefficients = beta_gls

    def get_params(self):
        # Get model parameters
        return pd.Series(self.coefficients, name='Beta coefficients')

    def get_pvalues(self):
        # Calculate p-values for coefficients
        self.fit()
        degrees_of_freedom = len(self.observed_values) - self.design_matrix.shape[1]
        residuals = self.observed_values - self.design_matrix @ self.coefficients
        residual_var = (residuals @ residuals) / degrees_of_freedom
        t_stat = self.coefficients / np.sqrt(np.diag(residual_var * np.linalg.inv(self.design_matrix.T @ self.weights_inv @ self.design_matrix)))

        # Calculate two-tailed p-values using the t distribution
        p_values = pd.Series([min(value, 1 - value) * 2 for value in t.cdf(-np.abs(t_stat), df=degrees_of_freedom)],
                             name='P-values for the corresponding coefficients')
        return p_values

    def get_wald_test_result(self, constraints_matrix):
        # Perform Wald test
        self.fit()
        r_matrix = np.array(constraints_matrix)
        r_transformed = r_matrix @ self.coefficients
        n_obs = len(self.observed_values)
        m_constraints, k_predictors = r_matrix.shape
        residuals = self.observed_values - self.design_matrix @ self.coefficients
        residual_var = (residuals @ residuals) / (n_obs - k_predictors)
        h_matrix = r_matrix @ np.linalg.inv(self.design_matrix.T @ self.weights_inv @ self.design_matrix) @ r_matrix.T
        wald_statistic = (r_transformed.T @ np.linalg.inv(h_matrix) @ r_transformed) / (m_constraints * residual_var)

        # Calculate p-value for the Wald statistic using the F distribution
        p_value = 1 - f_dist.cdf(wald_statistic, dfn=m_constraints, dfd=n_obs - k_predictors)
        return f'Wald: {wald_statistic:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        # Calculate goodness-of-fit measures
        self.fit()
        total_sos = self.observed_values.T @ self.weights_inv @ self.observed_values
        residual_sos = self.observed_values.T @ self.weights_inv @ self.design_matrix @ np.linalg.inv(self.design_matrix.T @ self.weights_inv @ self.design_matrix) @ self.design_matrix.T @ self.weights_inv @ self.observed_values
        centered_r_squared_new = 1 - (residual_sos / total_sos)
        adjusted_r_squared_new = 1 - (residual_sos / (len(self.observed_values) - self.design_matrix.shape[1])) * (
                len(self.observed_values) - 1) / total_sos

        return f"Centered R-squared: {centered_r_squared_new:.3f}, Adjusted R-squared: {adjusted_r_squared_new:.3f}"

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm  # Add this line
from scipy.optimize import minimize

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.optimize import minimize
from pytest import approx

class LinearRegressionML:
    def __init__(self, left_hand_side: pd.DataFrame, right_hand_side: pd.DataFrame):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.coefficients = None

    def _negative_log_likelihood(self, params: np.ndarray) -> float:
        y = self.left_hand_side.values.flatten()
        X = np.column_stack((np.ones_like(y), self.right_hand_side.values))
        mu = np.dot(X, params)
        neg_log_likelihood = -np.sum(norm.logpdf(y, loc=mu))
        return neg_log_likelihood

    def fit(self) -> None:
        initial_guess = 0.1 * np.ones(self.right_hand_side.shape[1] + 1)
        result = minimize(self._negative_log_likelihood, initial_guess, method='L-BFGS-B')
        self.coefficients = result.x

    def get_params(self, include_intercept: Optional[bool] = True) -> pd.Series:
        if self.coefficients is None:
            raise ValueError("Model has not been fitted. Please run the fit method.")

        param_names = ['Intercept' if include_intercept else '']
        param_names += [f'Beta_{i}' for i in range(1, len(self.coefficients))]

        return pd.Series(self.coefficients, index=param_names)


    def get_pvalues(self):
        self.fit()
        a, b = self.left_hand_side.shape[0], len(self.coefficients)
        X = np.column_stack((np.ones_like(self.left_hand_side), self.right_hand_side.values))
        residuals = self.left_hand_side.values.flatten() - np.dot(X, self.coefficients)
        sq = np.sum(residuals ** 2) / (a - b)
        beta = np.linalg.inv(np.dot(X.T, X)) * sq
        t_stat = self.coefficients / np.sqrt(np.diag(beta))
        p_values = 2 * (1 - t.cdf(np.abs(t_stat), df=a - b))
        return pd.Series(p_values, name='P-values for the corresponding coefficients')


    def get_model_goodness_values(self) -> str:
        if self.coefficients is None:
            raise ValueError("Model has not been fitted. Please run the fit method.")

        y = self.left_hand_side.values.flatten()
        X = np.column_stack((np.ones_like(y), self.right_hand_side.values))
        y_mean = np.mean(y)
        y_hat = np.dot(X, self.coefficients)
        centered_r_squared = 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y_mean) ** 2)

        # Calculate adjusted R-squared
        n, k = self.left_hand_side.shape[0], len(self.coefficients)
        adjusted_r_squared = 1 - ((1 - centered_r_squared) * (n - 1)) / (n - k)

        return f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"









