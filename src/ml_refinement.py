import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class MLRefinementLayer:
    def __init__(self, random_state=42):
        self.emulator = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.bias_corrector = RandomForestRegressor(n_estimators=100, random_state=random_state)

    def prepare_features(self, df):
        df_encoded = df.copy()
        date_mapping = {"10-Mar": 0, "20-Mar": 1, "05-Apr": 2, "20-Apr": 3}
        df_encoded["sowing_code"] = df_encoded["sowing_date"].map(lambda x: date_mapping.get(x, 1))
        return df_encoded

    def train_and_evaluate(self, df):
        df_encoded = self.prepare_features(df)
        X = df_encoded[["rainfall_mm", "nitrogen_kg_ha", "soc_pct", "sowing_code"]]
        y_sim = df_encoded["simulated_yield_kg_ha"]
        y_obs = df_encoded["observed_yield_kg_ha"]

        X_train, X_test, y_sim_tr, y_sim_te, y_obs_tr, y_obs_te = train_test_split(
            X, y_sim, y_obs, test_size=0.2, random_state=42
        )

        self.emulator.fit(X_train, y_sim_tr)
        y_sim_pred = self.emulator.predict(X_test)

        X_bc_tr = X_train.copy()
        X_bc_tr["simulated_yield"] = y_sim_tr
        X_bc_te = X_test.copy()
        X_bc_te["simulated_yield"] = y_sim_te

        self.bias_corrector.fit(X_bc_tr, y_obs_tr)
        y_obs_pred = self.bias_corrector.predict(X_bc_te)

        def compute_metrics(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mbe = np.mean(y_pred - y_true)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return {
                "RMSE (kg/ha)": round(rmse, 2),
                "MAE (kg/ha)": round(mae, 2),
                "R² Score": round(r2, 3),
                "MBE (kg/ha)": round(mbe, 2),
                "MAPE (%)": round(mape, 2),
            }

        metrics_df = pd.DataFrame(
            [compute_metrics(y_sim_te, y_sim_pred), compute_metrics(y_obs_te, y_obs_pred)],
            index=["Option A: ML Emulator (vs APSIM Sim)", "Option B: Bias Corrector (vs Field Obs)"],
        )

        return metrics_df, (X_test, y_sim_te, y_sim_pred, y_obs_te, y_obs_pred)