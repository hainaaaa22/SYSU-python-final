import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

if not os.path.exists("outputs"):
    os.mkdir("outputs")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------
# 1. 构建数据集（区域+时段 → 出行需求量）
# ---------------------------------------------------------------------
def build_demand_dataset(df):
    demand_df = df.groupby(['PULocationID', 'hour']).agg(
        demand=('hour', 'count'),
        day_of_week=('day_of_week', 'mean'),
        is_peak=('is_peak', 'mean')
    ).reset_index()
    X = demand_df[['PULocationID', 'hour', 'day_of_week', 'is_peak']].values
    y = demand_df['demand'].values
    return X, y, demand_df


# ---------------------------------------------------------------------
# 2. 8:2 划分数据集 + 【关键】标签标准化
# ---------------------------------------------------------------------
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # 特征标准化
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # 标签标准化（关键修复！）
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# ---------------------------------------------------------------------
# 3. 神经网络模型（增强版）
# ---------------------------------------------------------------------
class DemandNet(nn.Module):
    def __init__(self, input_dim):
        super(DemandNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ---------------------------------------------------------------------
# 4. 训练神经网络（修复版）
# ---------------------------------------------------------------------
def train_neural_network(X_train, y_train):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    model = DemandNet(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)  # 提高学习率

    epochs = 100  # 增加训练轮次
    loss_history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    history = {"loss": loss_history}
    return model, history


# ---------------------------------------------------------------------
# 5. 训练随机森林
# ---------------------------------------------------------------------
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------
# 6. 评估 + 打印预测结果
# ---------------------------------------------------------------------
def evaluate_and_show_results(nn_model, rf_model, X_test, y_test, demand_df, scaler_y):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    nn_model.eval()

    with torch.no_grad():
        y_pred_nn = nn_model(X_test_tensor).numpy().flatten()
    y_pred_rf = rf_model.predict(X_test)

    # 反标准化，还原成原始订单量
    y_true_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_nn_original = scaler_y.inverse_transform(y_pred_nn.reshape(-1, 1)).flatten()
    y_pred_rf_original = scaler_y.inverse_transform(y_pred_rf.reshape(-1, 1)).flatten()

    mae_nn = mean_absolute_error(y_true_original, y_pred_nn_original)
    rmse_nn = np.sqrt(mean_squared_error(y_true_original, y_pred_nn_original))
    mae_rf = mean_absolute_error(y_true_original, y_pred_rf_original)
    rmse_rf = np.sqrt(mean_squared_error(y_true_original, y_pred_rf_original))

    # 打印表格
    print("\n" + "=" * 85)
    print(" 区域ID | 时段 | 真实订单量 | 神经网络预测 | 随机森林预测 ")
    print("=" * 85)
    for i in range(min(15, len(y_true_original))):
        region = int(demand_df.iloc[i]['PULocationID'])
        hour = int(demand_df.iloc[i]['hour'])
        real = int(y_true_original[i])
        p_nn = round(y_pred_nn_original[i])
        p_rf = round(y_pred_rf_original[i])
        print(f"  {region:>3}   |  {hour:>2}  |    {real:>3}     |     {p_nn:>3}      |     {p_rf:>3}")
    print("=" * 85)

    print(f"\n📊 模型指标")
    print(f"神经网络 → MAE: {mae_nn:.2f} | RMSE: {rmse_nn:.2f}")
    print(f"随机森林 → MAE: {mae_rf:.2f} | RMSE: {rmse_rf:.2f}")

    return y_true_original, y_pred_nn_original, y_pred_rf_original, mae_nn, rmse_nn, mae_rf, rmse_rf


# ---------------------------------------------------------------------
# 7. 绘制对比图
# ---------------------------------------------------------------------
def plot_prediction_comparison(y_true, y_pred_nn, y_pred_rf):
    sample_size = min(30, len(y_true))
    x = np.arange(sample_size)

    plt.figure(figsize=(14, 6))
    plt.plot(x, y_true[:sample_size], marker='o', label="真实订单量", linewidth=2, color="#1f77b4")
    plt.plot(x, y_pred_nn[:sample_size], marker='s', label="神经网络预测", linewidth=2, color="#ff7f0e")
    plt.plot(x, y_pred_rf[:sample_size], marker='^', label="随机森林预测", linewidth=2, color="#2ca02c")

    plt.title("出行需求量预测对比（真实值 vs 神经网络 vs 随机森林）", fontsize=14)
    plt.xlabel("测试集样本序号", fontsize=12)
    plt.ylabel("订单需求量", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = "outputs/m3_prediction_comparison.png"
    plt.savefig(save_path, dpi=200)
    plt.show()
    return save_path


# ---------------------------------------------------------------------
# 8. 绘制 loss 曲线
# ---------------------------------------------------------------------
def plot_loss_curve(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='训练 Loss')
    plt.title('神经网络训练 Loss 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    save_path = "outputs/m3_loss_curve.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    return save_path


# ---------------------------------------------------------------------
# M3 主运行函数
# ---------------------------------------------------------------------
def run_m3_pipeline(df):
    print("\n===== M3 出行需求量预测模型（修复版）=====")

    X, y, demand_df = build_demand_dataset(df)
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = split_train_test(X, y)

    nn_model, history = train_neural_network(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    y_true_original, y_pred_nn_original, y_pred_rf_original, mae_nn, rmse_nn, mae_rf, rmse_rf = evaluate_and_show_results(
        nn_model, rf_model, X_test, y_test, demand_df, scaler_y
    )

    plot_prediction_comparison(y_true_original, y_pred_nn_original, y_pred_rf_original)
    plot_loss_curve(history)

    print("\n✅ M3 全部完成！对比图已保存到 outputs 文件夹")

    return {
        "nn_mae": mae_nn,
        "nn_rmse": rmse_nn,
        "rf_mae": mae_rf,
        "rf_rmse": rmse_rf
    }