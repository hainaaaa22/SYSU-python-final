from modules.m1_data import run_m1_pipeline
from modules.m2_visualize import run_m2_pipeline
from modules.m3_model import run_m3_pipeline

# 1. M1
df_final, report = run_m1_pipeline("D:\\35120\\中大\\python作业\\期末大作业\\yellow_tripdata_2023-01.parquet")

# 2. M2
run_m2_pipeline(df_final)

# 3. M3
m3_result = run_m3_pipeline(df_final)
print(m3_result["report"])