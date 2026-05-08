from modules.m1_data import run_m1_pipeline
from modules.m2_visualize import run_m2_pipeline
from modules.m3_model import run_m3_pipeline

# 1. 运行 M1
df_final, report = run_m1_pipeline("D:\\35120\\中大\\python作业\\期末大作业\\yellow_tripdata_2023-01.parquet")  # 记得改路径

# 2. 运行 M2
run_m2_pipeline(df_final)

# 3. 运行 M3（已经修复报错）
m3_result = run_m3_pipeline(df_final)

# 4. 打印结果（正确写法，不再报错）
print("\n最终 M3 模型指标：")
print(f"神经网络 MAE: {m3_result['nn_mae']:.2f}")
print(f"神经网络 RMSE: {m3_result['nn_rmse']:.2f}")
print(f"随机森林 MAE: {m3_result['rf_mae']:.2f}")
print(f"随机森林 RMSE: {m3_result['rf_rmse']:.2f}")