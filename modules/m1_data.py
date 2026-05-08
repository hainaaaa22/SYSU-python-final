import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 按官方文档导入pyarrow.parquet
import pyarrow.parquet as pq

# ---------------------- M1 核心功能函数 ----------------------
def load_taxi_data(file_path):
    """
    【已按官方PDF修改】加载.parquet格式的出租车数据
    :param file_path: 文件路径
    :return: 原始数据DataFrame
    """
    # ===================== 修改点1：使用官方推荐的parquet读取方式 =====================
    table = pq.read_table(file_path)
    df = table.to_pandas()

    # --- 检验代码（可删除）---
    print("=" * 50)
    print("数据加载成功！")
    print(f"数据形状: {df.shape}")
    print("\n前5行数据:")
    print(df.head())
    print("\n数据类型:")
    print(df.dtypes)
    # --- 检验代码结束 ---

    return df


def data_quality_report(df):
    """
    生成数据质量报告：缺失值、异常值、基础统计
    :param df: 原始数据
    :return: 报告文本
    """
    report = {}
    report['缺失值统计'] = df.isnull().sum().to_dict()
    report['缺失率'] = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    report['基础统计'] = df.describe().to_dict()

    # --- 检验代码（可删除）---
    print("\n" + "=" * 50)
    print("数据质量报告")
    print("缺失值数量：", report['缺失值统计'])
    print("缺失率(%)：", report['缺失率'])
    # --- 检验代码结束 ---

    return report


def clean_data(df):
    """
    【已按数据字典严格修改】清洗数据：所有规则对齐官方PDF
    每一步都标注理由
    """
    df_clean = df.copy()

    # 清洗1：去除票价<=0的数据（业务逻辑：车费不可能为0或负数）
    df_clean = df_clean[df_clean['fare_amount'] > 0]

    # 清洗2：去除总费用为负数（数据字典要求金额必须合法）
    df_clean = df_clean[df_clean['total_amount'] > 0]

    # 清洗3：去除行程距离<=0的数据（无效行程）
    df_clean = df_clean[df_clean['trip_distance'] > 0]

    # 清洗4：去除乘客数异常（0人或超过6人，不符合纽约出租车规则）
    df_clean = df_clean[(df_clean['passenger_count'] > 0) &
                        (df_clean['passenger_count'] <= 6)]

    # ===================== 修改点2：按数据字典新增清洗规则 =====================
    # 清洗5：RatecodeID 必须是 1-6（99为无效，剔除）
    df_clean = df_clean[df_clean['RatecodeID'].between(1, 6)]

    # 清洗6：上下车区域ID 不能为0（无效区域）
    df_clean = df_clean[(df_clean['PULocationID'] > 0) &
                        (df_clean['DOLocationID'] > 0)]

    # 清洗7：store_and_fwd_flag 必须是 Y 或 N
    df_clean = df_clean[df_clean['store_and_fwd_flag'].isin(['Y', 'N'])]

    # --- 检验代码（可删除）---
    print("\n" + "=" * 50)
    print(f"清洗前数据量: {len(df)}")
    print(f"清洗后数据量: {len(df_clean)}")
    # --- 检验代码结束 ---

    return df_clean


def extract_time_features(df):
    """
    【已修改】从 上车/下车 时间提取完整特征
    包含：小时、星期、是否高峰、行程时长
    """
    df_time = df.copy()

    # 确保两个时间列都转为标准格式
    df_time['tpep_pickup_datetime'] = pd.to_datetime(df_time['tpep_pickup_datetime'])
    # ===================== 修改点3：补充处理下车时间 =====================
    df_time['tpep_dropoff_datetime'] = pd.to_datetime(df_time['tpep_dropoff_datetime'])

    # 基础时间特征
    df_time['hour'] = df_time['tpep_pickup_datetime'].dt.hour
    df_time['day_of_week'] = df_time['tpep_pickup_datetime'].dt.dayofweek
    df_time['is_peak'] = df_time['hour'].apply(lambda x: 1 if 7 <= x <= 9 or 17 <= x <= 19 else 0)

    # ===================== 修改点4：新增 行程时长(分钟) =====================
    df_time['trip_duration_min'] = (
            (df_time['tpep_dropoff_datetime'] - df_time['tpep_pickup_datetime']).dt.total_seconds() / 60
    ).round(2)

    # 剔除异常短/长行程
    df_time = df_time[(df_time['trip_duration_min'] > 0) & (df_time['trip_duration_min'] < 300)]

    # --- 检验代码（可删除）---
    print("\n" + "=" * 50)
    print("时间特征提取完成！新增列：hour, day_of_week, is_peak, trip_duration_min")
    print(df_time[['hour', 'day_of_week', 'is_peak', 'trip_duration_min']].head())
    # --- 检验代码结束 ---

    return df_time


def build_derived_features(df):
    """
    自定义2个以上衍生特征（作业要求）
    全部基于数据字典字段，合法合规
    """
    df_feat = df.copy()

    # 自定义特征1：单位距离价格（性价比）
    df_feat['price_per_mile'] = df_feat['fare_amount'] / df_feat['trip_distance']

    # 自定义特征2：行程时间段分类
    def get_time_period(hour):
        if 6 <= hour <= 11:
            return '早上'
        elif 12 <= hour <= 17:
            return '下午'
        elif 18 <= hour <= 23:
            return '晚上'
        else:
            return '凌晨'

    df_feat['time_period'] = df_feat['hour'].apply(get_time_period)

    # 自定义特征3：小费比例（更适合分析/可视化，加分项）
    df_feat['tip_rate'] = (df_feat['tip_amount'] / df_feat['fare_amount']).round(3)

    # --- 检验代码（可删除）---
    print("\n" + "=" * 50)
    print("衍生特征构建完成！")
    print(df_feat[['price_per_mile', 'time_period', 'tip_rate']].head())
    # --- 检验代码结束 ---

    return df_feat


# ---------------------- M1 总流水线 ----------------------
def run_m1_pipeline(file_path):
    """
    一键执行M1所有流程
    输出：清洗+特征工程后的最终数据集
    """
    print("开始执行 M1 数据处理模块...")

    df_raw = load_taxi_data(file_path)
    quality_report = data_quality_report(df_raw)
    df_clean = clean_data(df_raw)
    df_with_time = extract_time_features(df_clean)
    df_final = build_derived_features(df_with_time)

    print("\n✅ M1 模块全部执行完成！")
    print(f"最终数据集形状: {df_final.shape}")

    return df_final, quality_report
