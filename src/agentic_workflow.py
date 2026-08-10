import pandas as pd

class AgronomicAgent:
    def __init__(self, apsim_driver, ml_layer):
        self.driver = apsim_driver
        self.ml_layer = ml_layer

    def get_weather_data(self, location="Western_Kenya"):
        return {"location": location, "rainfall_mm": 580.0, "avg_temp_c": 23.5}

    def get_soil_data(self, location="Western_Kenya"):
        return {"location": location, "soil_type": "Clay Loam", "soc_pct": 1.4}

    def run_advisory_pipeline(self, location="Western_Kenya", target_n=80.0, sow_date="15-Mar"):
        weather = self.get_weather_data(location)
        soil = self.get_soil_data(location)

        sim_yield = self.driver.run_single_simulation(
            nitrogen_kg_ha=target_n, sowing_date=sow_date, rainfall_mm=weather["rainfall_mm"]
        )

        feature_df = pd.DataFrame([{
            "rainfall_mm": weather["rainfall_mm"],
            "nitrogen_kg_ha": target_n,
            "soc_pct": soil["soc_pct"],
            "sowing_code": 1
        }])
        emulated_yield = self.ml_layer.emulator.predict(feature_df)[0]

        return (
            f"===========================================================\n"
            f"       IITA AGRONOMIC DECISION SUPPORT SYSTEM (ADSS)       \n"
            f"===========================================================\n"
            f"• Location Target   : {location}\n"
            f"• Environment       : Rainfall = {weather['rainfall_mm']}mm | Soil SOC = {soil['soc_pct']}%\n"
            f"• Management        : N Rate = {target_n} kg/ha | Sowing = {sow_date}\n"
            f"-----------------------------------------------------------\n"
            f"• APSIM Process Model Prediction : {sim_yield:,.1f} kg/ha ({sim_yield/1000:.2f} t/ha)\n"
            f"• ML Emulator Surrogate Prediction: {emulated_yield:,.1f} kg/ha ({emulated_yield/1000:.2f} t/ha)\n"
            f"-----------------------------------------------------------\n"
            f"• Agronomic Action: Apply Nitrogen split (50% basal at planting, 50% top-dressed at V6 stage).\n"
            f"==========================================================="
        )