import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.optimize import differential_evolution


def optimize_sensor(i, surrogate, x_ref):
    idx_a = i
    idx_u = i + num_sensors

    def objective(z):
        x = x_ref.copy()
        x[idx_a] = z[0]      # a_i
        x[idx_u] = z[1]      # u_dc_i
        return -surrogate.predict(x.reshape(1, -1))[0]

    bounds = [
        (-25.0, 25.0),  # a_i
        (0.0, 1.0)      # u_dc_i
    ]

    result = differential_evolution(
        objective,
        bounds,
        strategy="best1bin",
        maxiter=150,
        popsize=20,
        tol=1e-4,
        seed=42
    )

    return {
        "sensor": i,
        "a_opt": result.x[0],
        "u_dc_opt": result.x[1],
        "predicted_value": -result.fun
    }


df = pd.read_csv("64_sens_optimization.csv")
df = df.dropna()

a_cols = sorted([c for c in df.columns if c.startswith("Param a_")])
u_cols = sorted([c for c in df.columns if c.startswith("Param u_dc_")])

assert len(a_cols) == len(u_cols)
num_sensors = len(a_cols)

X = df[a_cols + u_cols].values
y = df["Value"].values

surrogate = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

surrogate.fit(X, y)

best_row = df.loc[df["Value"].idxmax()]
x_ref = best_row[a_cols + u_cols].values.copy()


# results = []
# for i in range(num_sensors):
#     print(f"Optimizing sensor {i}")
#     res = optimize_sensor(i, surrogate, x_ref)
#     results.append(res)


results = Parallel(n_jobs=64, backend="multiprocessing", verbose=1)(delayed(optimize_sensor)(i, surrogate, x_ref) for i in range(num_sensors))

results = np.array(results)

opt_df = pd.DataFrame(results)

opt_df.to_csv("sensor_optimization_results.csv", index=False)

importances = surrogate.feature_importances_

np.save("sensor_importances.npy", importances)