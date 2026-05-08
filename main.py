from modules.m1_data import run_m1_pipeline, visualize_m1_results

df_final, report = run_m1_pipeline("D:\\35120\\中大\\python作业\\期末大作业\\yellow_tripdata_2023-01.parquet")
visualize_m1_results(df_final)