from modules.m1_data import run_m1_pipeline
from modules.m2_visualize import run_m2_pipeline

# 1. 运行M1
df_final, report = run_m1_pipeline("D:\\35120\\中大\\python作业\\期末大作业\\yellow_tripdata_2023-01.parquet")

# 2. 运行M2（直接用M1的结果）
image_paths = run_m2_pipeline(df_final)