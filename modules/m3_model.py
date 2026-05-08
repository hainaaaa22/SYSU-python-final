import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 用 PyTorch 构建神经网络（可安装成功版）
import torch
import torch.nn as nn
import torch.optim as optim

# 输出文件夹
if not os.path.exists("outputs"):
    os.mkdir("outputs")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------------------------------------------------------------------
# 1. 构建【区域+时段】出行需求量数据集
# ---------------------------------------------------------------------
def build_demand_dataset(df):
    demand_df = df.groupby(['PULocationID', 'hour']).agg(
        demand=('hour', 'count'),
        day_of_week=('day_of_week', 'mean'),
        is_peak=('is_peak', 'mean')
    ).reset_index()

    X = demand_df[['PULocationID', 'hour', 'day_of_week', 'is_peak']].values
    y = demand_df['demand'].values

    # --- 检验代码（可删除）---
    print("构建需求量数据集完成")
    print(f"数据集形状: {X.shape}")
    print(f"前5条数据:\n{demand_df.head()}")
    # --- 检验代码结束 ---

    return X, y, demand_df


# ---------------------------------------------------------------------
# 2. 8:2 划分训练集 / 测试集
# ---------------------------------------------------------------------
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# ---------------------------------------------------------------------
# PyTorch 神经网络模型
# ---------------------------------------------------------------------
class DemandNet(nn.Module):
    def __init__(self, input_dim):
        super(DemandNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------
# 3. 训练 PyTorch 神经网络
# ---------------------------------------------------------------------
def train_neural_network(X_train, y_train):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    model = DemandNet(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    history = {"loss": loss_history, "val_loss": val_loss_history}
    return model, history


# ---------------------------------------------------------------------
# 4. 训练对比模型：随机森林
# ---------------------------------------------------------------------
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------
# 5. 评估两个模型，输出 MAE、RMSE
# ---------------------------------------------------------------------
def evaluate_models(nn_model, rf_model, X_test, y_test):
    # 神经网络预测
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    nn_model.eval()
    with torch.no_grad():
        y_pred_nn = nn_model(X_test_tensor).numpy().flatten()

    # 随机森林预测
    y_pred_rf = rf_model.predict(X_test)

    # 指标
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    report = f"""
========== 模型对比结果 ==========
PyTorch 神经网络：
MAE = {mae_nn:.2f}
RMSE = {rmse_nn:.2f}

随机森林：
MAE = {mae_rf:.2f}
RMSE = {rmse_rf:.2f}
==================================
"""
    print(report)
    return report, mae_nn, rmse_nn, mae_rf, rmse_rf


# ---------------------------------------------------------------------
# 6. 绘制 loss 曲线
# ---------------------------------------------------------------------
def plot_loss_curve(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='训练loss')
    plt.title('神经网络训练 Loss 曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    save_path = "outputs/m3_loss_curve.png"
    plt.savefig(save_path, dpi=150)

    # --- 检验代码（可删除）---
    plt.show()
    # --- 检验代码结束 ---

    return save_path


# ---------------------------------------------------------------------
# M3 总流水线（一键运行）
# ---------------------------------------------------------------------
def run_m3_pipeline(df):
    print("开始执行 M3 预测模型模块...")

    X, y, demand_df = build_demand_dataset(df)
    X_train, X_test, y_train, y_test, scaler = split_train_test(X, y)

    nn_model, history = train_neural_network(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    report, mae_nn, rmse_nn, mae_rf, rmse_rf = evaluate_models(nn_model, rf_model, X_test, y_test)
    loss_img = plot_loss_curve(history)

    print("\n✅ M3 全部执行完成！")
    return {
        "nn_model": nn_model,
        "rf_model": rf_model,
        "scaler": scaler,
        "report": report,
        "loss_image": loss_img
    }