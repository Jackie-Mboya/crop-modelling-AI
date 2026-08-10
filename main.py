import os
from src.apsim_driver import NativeAPSIMDriver
from src.ml_refinement import MLRefinementLayer
from src.agentic_workflow import AgronomicAgent
from src.evaluate import generate_evaluation_plots

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("=========================================================")
    print("STEP 1 & 2: APSIM PROCESS MODEL SCENARIOS RUN")
    print("=========================================================")
    driver = NativeAPSIMDriver()
    step2_df = driver.run_step2_scenarios()
    print(step2_df.to_string(index=False))
    step2_df.to_csv("outputs/scenario_comparison.csv", index=False)

    print("\n=========================================================")
    print("STEP 3: GENERATE SIMULATION DATASET (n=100)")
    print("=========================================================")
    sim_df = driver.generate_simulation_dataset(n_runs=100)
    sim_df.to_csv("data/simulation_dataset.csv", index=False)
    print(sim_df.head().to_string(index=False))

    print("\n=========================================================")
    print("STEP 4 & 5: TRAIN ML LAYERS & EVALUATE METRICS")
    print("=========================================================")
    ml_layer = MLRefinementLayer()
    metrics_df, eval_tuple = ml_layer.train_and_evaluate(sim_df)
    print(metrics_df.to_string())
    metrics_df.to_csv("outputs/evaluation_metrics.csv")

    generate_evaluation_plots(eval_tuple, ml_layer.emulator)

    print("\n=========================================================")
    print("STEP 4 (Option D): AGENTIC AI WORKFLOW RUN")
    print("=========================================================")
    agent = AgronomicAgent(driver, ml_layer)
    advisory = agent.run_advisory_pipeline(location="Western Kenya", target_n=85.0, sow_date="15-Mar")
    print(advisory)

if __name__ == "__main__":
    main()