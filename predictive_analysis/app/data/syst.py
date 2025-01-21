import pandas as pd
import numpy as np

# Generate synthetic manufacturing data
np.random.seed(42)
data = {
    "Machine_ID": np.arange(1, 101),
    "Temperature": np.random.uniform(50, 100, 100),
    "Run_Time": np.random.uniform(100, 1000, 100),
    "Downtime_Flag": np.random.choice([0, 1], size=100, p=[0.8, 0.2]),
}

df = pd.DataFrame(data)
df.to_csv("manufacturing_data.csv", index=False)
print("Sample dataset saved as manufacturing_data.csv.")
