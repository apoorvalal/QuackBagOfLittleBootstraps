# %%
import time
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression


# %%
class BLB:
    """
    Reference class for Bag of Little Bootstraps (BLB) confidence intervals -
    in memory, parallelizable
    """

    def __init__(
        self,
        estimator: Callable[[np.ndarray], float],
        confidence_level: float = 0.95,
        s: int = 20,
        r: int = 50,
        b_power: float = 0.65,
        n_jobs: int = -1,
    ):
        self.estimator = estimator
        self.s = s
        self.r = r
        self.b_power = b_power
        self.alpha = 1 - confidence_level
        self.n_jobs = n_jobs

    def confidence_interval(self, data: np.ndarray) -> tuple:
        """Compute confidence intervals using parallel BLB"""
        n = len(data)
        b = int(n**self.b_power)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_subsample)(j, data, n, b) for j in range(self.s)
        )

        # Unzip and average along subsample dimension
        lower_assessments, upper_assessments = zip(*results)
        return (np.mean(lower_assessments, axis=0), np.mean(upper_assessments, axis=0))

    ######################################################################
    ######################################################################
    def _bootstrap_estimates(self, subsample: np.ndarray, n: int, b: int) -> np.ndarray:
        """Compute bootstrap estimates for a single subsample"""
        estimates = np.zeros(self.r)
        for k in range(self.r):
            boot_idx = np.random.choice(b, size=n, replace=True)
            boot_sample = subsample[boot_idx]
            estimates[k] = self.estimator(boot_sample)
        return estimates

    def _process_subsample(self, j: int, data: np.ndarray, n: int, b: int) -> tuple:
        """Process a single subsample"""
        subsample_idx = np.random.choice(n, size=b, replace=False)
        subsample = data[subsample_idx]

        # Get bootstrap estimates - now each estimate could be a vector
        estimates = np.zeros((self.r, *np.shape(self.estimator(subsample))))
        for k in range(self.r):
            boot_idx = np.random.choice(b, size=n, replace=True)
            boot_sample = subsample[boot_idx]
            estimates[k] = self.estimator(boot_sample)

        # Compute bounds along first axis (the bootstrap dimension)
        lower = np.percentile(estimates, self.alpha * 50, axis=0)
        upper = np.percentile(estimates, (2 - self.alpha) * 50, axis=0)

        return lower, upper


if __name__ == "__main__":
    # Generate some test data
    np.random.seed(42)

    def onesim(n_samples=1_000_000):
        X = np.random.normal(0, 1, (n_samples, 2))
        y = X.dot([1, 2]) + np.random.normal(0, 1, n_samples)
        data = np.column_stack([X, y])
        return data

    # For full coefficient vector:
    def coef_estimator(x):
        return LinearRegression().fit(x[:, :-1], x[:, -1]).coef_  # returns whole vector

    # Time parallel vs non-parallel
    for n_jobs in [1, -1]:  # 1 for serial, -1 for all cores
        data = onesim()
        blb = BLB(estimator=coef_estimator, n_jobs=n_jobs)
        start = time.time()
        ci = blb.confidence_interval(data)
        elapsed = time.time() - start
        print(f"\nUsing n_jobs={n_jobs}:")
        print(f"Time taken: {elapsed:.2f} seconds")
        print(np.c_[coef_estimator(data), ci])
