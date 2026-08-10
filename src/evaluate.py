import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_evaluation_plots(eval_tuple, emulator_model, output_path="outputs/evaluation_plots.png"):
    X_test, y_sim_te, y_sim_pred, y_obs_te, y_obs_pred = eval_tuple

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_sim_te, y_sim_pred, alpha=0.75, color="#1f77b4", edgecolors="k", linewidth=0.5, label="Test Scenarios")
    axes[0].plot([y_sim_te.min(), y_sim_te.max()], [y_sim_te.min(), y_sim_te.max()], "r--", lw=2, label="1:1 Parity Line")
    axes[0].set_title("ML Emulator vs APSIM Simulated Yield", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("APSIM Simulated Yield (kg/ha)")
    axes[0].set_ylabel("ML Predicted Yield (kg/ha)")
    axes[0].legend()

    n_range = np.linspace(0, 160, 50)
    curve_df = pd.DataFrame({"rainfall_mm": 580.0, "nitrogen_kg_ha": n_range, "soc_pct": 1.4, "sowing_code": 1})
    curve_pred = emulator_model.predict(curve_df)

    axes[1].plot(n_range, curve_pred, color="#2ca02c", lw=2.5, label="ML Response Curve (580mm Rain)")
    axes[1].scatter(X_test["nitrogen_kg_ha"], y_sim_pred, alpha=0.5, color="#ff7f0e", label="Test Points")
    axes[1].set_title("Maize Yield Response to Nitrogen Rate", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Nitrogen Applied (kg/ha)")
    axes[1].set_ylabel("Yield (kg/ha)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Evaluation plots saved to '{output_path}'.")