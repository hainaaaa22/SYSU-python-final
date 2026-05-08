from modules.m1_data import run_m1_pipeline
from modules.m3_model import run_m3_pipeline
from modules.m4_chat import start_chat_gui

if __name__ == "__main__":
    print("正在运行 M1 数据处理...")
    df_final, _ = run_m1_pipeline("./yellow_tripdata_2023-01.parquet")

    print("正在运行 M3 预测模型...")
    m3_result = run_m3_pipeline(df_final)

    # 启动窗口版问答，不再用命令行
    start_chat_gui(df_final, m3_result)