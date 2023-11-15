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



class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.beta = None  # Change to self.beta for consistency


    def fit(self):
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        y = self.left_hand_side
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.beta = beta

    def get_params(self):
        return pd.Series(self.beta, name='Beta coefficients')

    def get_pvalues(self):
        if self.beta_coefficients is None:
            raise ValueError("A modell még nem lett becslve. Kérlek futtasd el a fit metódust először.")

        # Compute t-statistic and p-values
        X = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side))
        residuals = self.left_hand_side - X @ self.beta_coefficients
        squared_residuals = residuals**2
        X_squared = np.column_stack((np.ones(len(self.right_hand_side)), self.right_hand_side, squared_residuals))
        t_statistic = self.beta_coefficients / np.sqrt(np.diag(np.linalg.inv(X_squared.T @ X_squared)))
        p_values = pd.Series(np.minimum(t_statistic, 1 - t_statistic) * 2, index=range(len(self.beta_coefficients)))
        p_values.name = 'P-values for the corresponding coefficients'
        return p_values

    def get_wald_test_result(self, R):
        if self.beta_coefficients is None:
            raise ValueError("A modell még nem lett becslve. Kérlek futtasd el a fit metódust először.")
        wald_value = float((R @ self.beta_coefficients).T @ np.linalg.inv(R @ np.linalg.inv(X.T @ weights @ X) @ R.T) @ (R @ self.beta_coefficients))
        p_value = 1 - stats.chi2.cdf(wald_value, len(R))
        result_string = f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"
        return result_string


    def get_model_goodness_values(self):
        if self.beta_coefficients is None:
            raise ValueError("A modell még nem lett becslve. Kérlek futtasd el a fit metódust először.")
        residuals = self.left_hand_side - X @ self.beta_coefficients.values
        SSR = np.sum(residuals**2)
        mean_response = np.mean(self.left_hand_side)
        SST = np.sum((self.left_hand_side - mean_response)**2)

        crs = 1 - SSR / SST
        n = len(self.left_hand_side)
        k = X.shape[1]  # A változók száma (beleértve a konstanst is)
        ars = 1 - (1 - crs) * (n - 1) / (n - k - 1)

        result_string = f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"

        return result_string


