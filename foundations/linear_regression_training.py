import numpy as np
from numpy.typing import NDArray

class Solution:
    # We keep the original methods intact so the class structure matches your requirements
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        
        # Take a copy to avoid mutating the original input reference directly
        weights = initial_weights.copy()
        N = len(X)

        for _ in range(num_iterations):
            # 1. Compute predictions for all samples at once
            y_pred = X @ weights  # Much cleaner alternative to np.squeeze(np.matmul())
            
            # 2 & 3. COMPLETELY OPTIMIZED: Eliminate the inner 'j' loop entirely!
            # (X.T @ (y_pred - Y)) calculates the gradient for EVERY single weight instantly.
            gradients = (2 / N) * (X.T @ (y_pred - Y))
            
            # Update all weights simultaneously
            weights -= self.learning_rate * gradients

        return np.round(weights, 5)