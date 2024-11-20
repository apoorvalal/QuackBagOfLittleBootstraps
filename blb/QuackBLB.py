# %%
from typing import Callable, Tuple

import duckdb
import numpy as np


# %%
class QuackBLB:
    def __init__(
        self,
        estimator: Callable[[np.ndarray], np.ndarray],
        confidence_level: float = 0.95,
        s: int = 20,
        r: int = 100,
        b_power: float = 0.7,
    ):
        self.estimator = estimator
        self.s = s
        self.r = r
        self.b_power = b_power
        self.alpha = 1 - confidence_level
        self.subsample_query = None

    def confidence_interval(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        n: int,  # total rows in source table
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence intervals using parallel BLB

        Args:
            conn: DuckDB connection
            query: SQL query for source data
            n: Total number of rows in source data
        """
        self.b = int(n**self.b_power)
        self.fraction = self.b / n

        def process_subsample(j: int) -> Tuple[np.ndarray, np.ndarray]:
            # Get subsample from DuckDB
            subsample = self._get_subsample(conn, query, self.fraction, seed=j)
            # Compute bootstrap on subsample
            return self._bootstrap_compute(subsample, n)

        # Parallel processing of subsamples
        results = [process_subsample(j) for j in range(self.s)]

        # Average bounds across subsamples
        lower_assessments, upper_assessments = zip(*results)
        # return lower_assessments, upper_assessments
        return (np.mean(lower_assessments, axis=0), np.mean(upper_assessments, axis=0))

    ######################################################################
    # layer 1 - subsample in DuckDB
    ######################################################################
    def _get_subsample(
        self, conn: duckdb.DuckDBPyConnection, query: str, fraction: float, seed: int
    ) -> np.ndarray:
        """Draw subsample from DuckDB"""
        self.subsample_query = f"""
        SELECT *
            FROM ({query})
            USING SAMPLE {fraction * 100: .4f}%
            (system, {seed})
        """
        dfsamp = conn.execute(self.subsample_query).fetchdf().values
        return dfsamp

    ######################################################################
    # layer 2 - bootstrap in memory
    ######################################################################
    def _bootstrap_compute(
        self, subsample: np.ndarray, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bootstrap estimates given a subsample"""
        b = len(subsample)
        estimates = np.zeros((self.r, *np.shape(self.estimator(subsample))))

        for k in range(self.r):
            boot_idx = np.random.choice(b, size=n, replace=True)
            boot_sample = subsample[boot_idx]
            estimates[k] = self.estimator(boot_sample)

        lower = np.percentile(estimates, self.alpha * 50, axis=0)
        upper = np.percentile(estimates, (2 - self.alpha) * 50, axis=0)
        return lower, upper


