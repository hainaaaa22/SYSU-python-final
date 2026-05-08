import pandas as pd
import matplotlib.pyplot as plt
import os

# 创建保存图片的文件夹（自动）
if not os.path.exists("outputs"):
    os.mkdir("outputs")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------
# 1. 出行需求时间规律分析
# ---------------------------------------------------------------------
def plot_time_pattern(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 小时订单量
    hour_count = df['hour'].value_counts().sort_index()
    axes[0].bar(hour_count.index, hour_count.values, color='#42A5F5')
    axes[0].set_title('各小时出行订单量')
    axes[0].set_xlabel('小时')
    axes[0].set_ylabel('订单数')

    # 高峰 vs 非高峰
    peak_labels = {0: '非高峰', 1: '高峰'}
    peak_count = df['is_peak'].map(peak_labels).value_counts()
    axes[1].pie(peak_count, labels=peak_count.index, autopct='%1.1f%%', colors=['#FFA726', '#66BB6A'])
    axes[1].set_title('高峰/非高峰订单占比')

    plt.tight_layout()
    save_path = "outputs/m2_time_pattern.png"
    plt.savefig(save_path, dpi=150)


    return save_path


# ---------------------------------------------------------------------
# 2. 区域热度 TOP10 上下车区域
# ---------------------------------------------------------------------
def plot_zone_hotmap(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 上车TOP10
    pu_top10 = df['PULocationID'].value_counts().head(10)
    axes[0].barh(pu_top10.index.astype(str), pu_top10.values, color='#AB47BC')
    axes[0].set_title('上车区域 TOP10')
    axes[0].set_xlabel('订单量')

    # 下车TOP10
    do_top10 = df['DOLocationID'].value_counts().head(10)
    axes[1].barh(do_top10.index.astype(str), do_top10.values, color='#5C6BC0')
    axes[1].set_title('下车区域 TOP10')
    axes[1].set_xlabel('订单量')

    plt.tight_layout()
    save_path = "outputs/m2_zone_top10.png"
    plt.savefig(save_path, dpi=150)

    return save_path


# ---------------------------------------------------------------------
# 3. 车费影响因素分析
# ---------------------------------------------------------------------
def plot_fare_factors(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 距离 vs 车费
    sample = df.sample(min(5000, len(df)))  # 抽样避免太密
    axes[0].scatter(sample['trip_distance'], sample['fare_amount'], alpha=0.4, color='#26A69A')
    axes[0].set_title('行程距离 vs 车费')
    axes[0].set_xlabel('距离')
    axes[0].set_ylabel('车费')

    # 时段平均车费
    period_fare = df.groupby('time_period')['fare_amount'].mean()
    axes[1].bar(period_fare.index, period_fare.values, color='#FF7043')
    axes[1].set_title('不同时段平均车费')

    plt.tight_layout()
    save_path = "outputs/m2_fare_factors.png"
    plt.savefig(save_path, dpi=150)


    return save_path


# ---------------------------------------------------------------------
# 4. 自选分析：不同支付方式的小费差异（你选的A）
# ---------------------------------------------------------------------
def plot_payment_tip_analysis(df):
    # 只保留有意义的数据
    sub = df[(df['tip_amount'] >= 0) & (df['payment_type'].isin([1, 2]))].copy()

    # 映射支付方式
    payment_map = {1: "信用卡", 2: "现金"}
    sub['payment_type_name'] = sub['payment_type'].map(payment_map)

    # 计算平均小费
    tip_mean = sub.groupby('payment_type_name')['tip_amount'].mean()

    plt.figure(figsize=(8, 5))
    tip_mean.plot(kind='bar', color=['#42A5F5', '#FFA726'])
    plt.title('不同支付方式平均小费对比')
    plt.ylabel('平均小费')
    plt.tight_layout()

    save_path = "outputs/m2_payment_tip.png"
    plt.savefig(save_path, dpi=150)


    return save_path


# ---------------------------------------------------------------------
# M2 统一运行入口（一键执行所有）
# ---------------------------------------------------------------------
def run_m2_pipeline(df):
    print("开始执行 M2 可视化模块...")

    paths = []
    paths.append(plot_time_pattern(df))
    paths.append(plot_zone_hotmap(df))
    paths.append(plot_fare_factors(df))
    paths.append(plot_payment_tip_analysis(df))

    print("✅ M2 全部图表生成完成！")
    for p in paths:
        print(f"保存：{p}")

    return paths