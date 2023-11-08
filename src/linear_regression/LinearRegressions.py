
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
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

import pandas as pd
import numpy as np
from scipy.stats import t, f
from typing import List

class LinearRegressionNP:
    def __init__(self, left_hand_side: pd.DataFrame, right_hand_side: pd.DataFrame):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.coefficients = None

    def fit(self):
        X = self.right_hand_side[['Mkt-RF', 'SMB', 'HML']].values
        X = np.column_stack((np.ones(X.shape[0]), X))
        y = self.left_hand_side['Excess Return'].values
        beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        self.coefficients = beta

    def get_params(self) -> pd.Series:
        if self.coefficients is not None:
            param_names = ['Alpha', 'Mkt-RF', 'SMB', 'HML']
            param_values = [self.coefficients[0]] + list(self.coefficients[1:])
            beta_params = pd.Series(param_values, index=param_names, name="Beta coefficients")
            return beta_params
        else:
            raise ValueError("Model coefficients are not available. Fit the model first.")

    def __get_pvalues__(self) -> pd.Series:
        if self.coefficients is not None:
            X = self.right_hand_side[['Mkt-RF', 'SMB', 'HML']].values
            X = np.column_stack((np.ones(X.shape[0]), X))
            y = self.left_hand_side['Excess Return'].values
            n, k = X.shape[0], X.shape[1

            y_hat = X @ self.coefficients
            residuals = y - y_hat

            # Hiba négyzetösszeg
            SSE = residuals @ residuals

            # Hiba szórásnégyzet
            MSE = SSE / (n - k)

            # Béta együtthatók standard hibái
            XTX_inv = np.linalg.inv(X.T @ X)
            beta_std_err = np.sqrt(np.diagonal(MSE * XTX_inv))

            # t-statisztika
            t_stats = self.coefficients / beta_std_err

            # P-érték kiszámítása a t-statisztika alapján
            p_values = (1 - t.cdf(np.abs(t_stats), df=n - k)) * 2

            param_names = ['Alpha', 'Mkt-RF', 'SMB', 'HML']
            p_values_series = pd.Series(p_values, index=param_names, name="P-values for the corresponding coefficients")
            return p_values_series
        else:
            raise ValueError("Model coefficients are not available. Fit the model first.")

    def __get_wald_test_result__(self, R: List[float]) -> str:
        if self.coefficients is not None:
            X = self.right_hand_side[['Mkt-RF', 'SMB', 'HML']].values
            X = np.column_stack((np.ones(X.shape[0]), X))
            y = self.left_hand_side['Excess Return'].values
            n, k = X.shape[0], X.shape[1]

            # Becsült hibák
            y_hat = X @ self.coefficients
            residuals = y - y_hat

            # Hiba négyzetösszeg
            SSE = residuals @ residuals

            # Hiba szórásnégyzet
            MSE = SSE / (n - k)

            # Béta együtthatók standard hibái
            XTX_inv = np.linalg.inv(X.T @ X)
            beta_std_err = np.sqrt(np.diagonal(MSE * XTX_inv))

            # t-statisztika
            t_stats = self.coefficients / beta_std_err

            # P-érték kiszámítása a t-statisztika alapján
            p_values = (1 - t.cdf(np.abs(t_stats), df=n - k)) * 2

            # Wald statisztika számítása
            R = np.array(R)
            wald_value = (R @ self.coefficients) @ np.linalg.inv(R @ XTX_inv @ R.T) @ R @ self.coefficients
            wald_value = wald_value.item()

            # Wald statisztika p-értékének számítása
            p_value = 1 - f.cdf(wald_value, len(R), n - k)

            return f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"
        else:
            raise ValueError("Model coefficients are not available. Fit the model first.")

    def __get_model_goodness_values__(self) -> str:
        if self.coefficients is not None:
            X = self.right_hand_side[['Mkt-RF', 'SMB', 'HML']].values
            X = np.column_stack((np.ones(X.shape[0]), X))
            y = self.left_hand_side['Excess Return'].values
            n, k = X.shape[0], X.shape[1

            # Becsült hibák
            y_hat = X @ self.coefficients
            residuals = y - y_hat

            # Hiba négyzetösszeg
            SSE = residuals @ residuals

            # Teljes négyzetösszeg
            SST = (y - np.mean(y)) @ (y - np.mean(y))

            # Centrált R-négyzet
            crs = 1 - SSE / SST

            # Módosított R-négyzet
            ars = 1 - (SSE / (n - k - 1)) / (SST / (n - 1))

            return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"
        else:
            raise ValueError("Model coefficients are not available. Fit the model first.")















