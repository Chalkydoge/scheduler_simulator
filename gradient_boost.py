import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import joblib


class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_and_process_data(self):
        # 1. 数据加载
        data = pd.read_csv(self.filepath)
        print(data.head())

        # 过滤掉 replica_num == 0 的数据
        # filtered_data = data[data['replica_num'] != 0]

        # 2. 数据预处理
        X = data[['miss', 'cache_ref', 'instruction', 'ipc']]
        Y = data[['transfer', 'bitrate']]

        # 处理缺失值
        _imp = SimpleImputer(strategy='mean')
        x_imputed = _imp.fit_transform(X)
        y_imputed = _imp.fit_transform(Y)

        # 标准化 去掉因为后面还需要推理 输入的x是原始数据
        # scaler = StandardScaler()
        # x_scaled = scaler.fit_transform(x_imputed)
        return x_imputed, y_imputed


class GradientBoostModel:
    def __init__(self):
        self.model_transfer = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3,
                                                        random_state=42)
        self.model_bitrate = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3,
                                                       random_state=42)
        self.model_transfer_path = 'model_transfer.joblib'
        self.model_bitrate_path = 'model_bitrate.joblib'

    def train(self, x_train, y_train):
        # y_train的第一个维度对应于传输率
        self.model_transfer.fit(x_train, y_train[:, 0])
        # 第二个维度对应于比特率
        self.model_bitrate.fit(x_train, y_train[:, 1])

    def save_model(self, filepath_transfer, filepath_bitrate):
        joblib.dump(self.model_transfer, filepath_transfer)
        joblib.dump(self.model_bitrate, filepath_bitrate)
        # print(f"Models saved to {filepath_transfer} and {filepath_bitrate}")

    def load_model(self, filepath_transfer, filepath_bitrate):
        self.model_transfer = joblib.load(filepath_transfer)
        self.model_bitrate = joblib.load(filepath_bitrate)
        # print(f"Models loaded from {filepath_transfer} and {filepath_bitrate}")

    def predict(self, x):
        y_pred_transfer = self.model_transfer.predict(x)
        y_pred_bitrate = self.model_bitrate.predict(x)
        return y_pred_transfer, y_pred_bitrate


def predict_plot(model_transfer, model_bitrate, X, Y):
    Y_pred_transfer, Y_pred_bitrate = model_transfer.predict(X), model_bitrate.predict(X)

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=Y[:, 0], y=Y_pred_transfer, color='blue')
    plt.plot([Y[:, 0].min(), Y[:, 0].max()], [Y[:, 0].min(), Y[:, 0].max()], 'r--', lw=2)
    plt.title('Transfer Rate: Predicted vs Actual')
    plt.xlabel('Actual Transfer Rate')
    plt.ylabel('Predicted Transfer Rate')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=Y[:, 1], y=Y_pred_bitrate, color='green')
    plt.plot([Y[:, 1].min(), Y[:, 1].max()], [Y[:, 1].min(), Y[:, 1].max()], 'r--', lw=2)
    plt.title('Bitrate: Predicted vs Actual')
    plt.xlabel('Actual Bitrate')
    plt.ylabel('Predicted Bitrate')

    plt.tight_layout()
    plt.savefig('gb1.png')


def y_compare(ytest, ypred):
    data = pd.read_csv('./data/cnf_update.csv')
    replica_num = data['replica_num']
    mask = replica_num != 0
    filtered_replica_num = replica_num[mask]
    # 这两个长度是一样的
    filtered_actual_transfer = ytest[:, 0]
    filtered_predicted_transfer = ypred

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(filtered_replica_num, filtered_actual_transfer, label='Actual Transfer', color='blue', marker='o',
            linestyle='-', markersize=4)
    ax.plot(filtered_replica_num, filtered_predicted_transfer, label='Predicted Transfer', color='red', linestyle='--',
            marker='x', markersize=4)
    ax.set_title('Transfer Rate: Actual vs Predicted by Replica Number (Excluding replica_num=0)')
    ax.set_xlabel('Replica Number')
    ax.set_ylabel('Transfer Rate')
    ax.legend()

    axins = inset_axes(ax, width="30%", height="30%", loc='upper left', borderpad=2)
    axins.plot(filtered_replica_num, filtered_actual_transfer, label='Actual Transfer', color='blue', marker='o',
               linestyle='-', markersize=4)
    axins.plot(filtered_replica_num, filtered_predicted_transfer, label='Predicted Transfer', color='red',
               linestyle='--', marker='x', markersize=4)
    axins.set_xlim(0, 20)
    axins.set_ylim(min(filtered_actual_transfer[filtered_replica_num <= 20].min(),
                       filtered_predicted_transfer[filtered_replica_num <= 20].min()) - 0.1,
                   max(filtered_actual_transfer[filtered_replica_num <= 20].max(),
                       filtered_predicted_transfer[filtered_replica_num <= 20].max()) + 0.1)
    axins.set_xticks([0, 5, 10, 15, 20])
    axins.set_yticks([min(filtered_actual_transfer[filtered_replica_num <= 20].min(),
                          filtered_predicted_transfer[filtered_replica_num <= 20].min()) - 0.1,
                      (max(filtered_actual_transfer[filtered_replica_num <= 20].max(),
                           filtered_predicted_transfer[filtered_replica_num <= 20].max()) +
                       min(filtered_actual_transfer[filtered_replica_num <= 20].min(),
                           filtered_predicted_transfer[filtered_replica_num <= 20].min())) / 2,
                      max(filtered_actual_transfer[filtered_replica_num <= 20].max(),
                          filtered_predicted_transfer[filtered_replica_num <= 20].max()) + 0.1])
    ax.indicate_inset_zoom(axins)
    plt.savefig('transfer_rate_comparison_zoomed_no_0.png', dpi=300)
    plt.show()


def calculate_degradation(ypred):
    # 获取 replica_num=1 的基准 transfer 值
    data = pd.read_csv('./data/cnf_update.csv')
    replica_num = data['replica_num']
    df = data[replica_num == 1]
    baseline_transfer = (df['transfer'].mean())
    # 计算每个 replica_num 的 transfer 相对于基准值的比值
    degradation_ratios = ypred / baseline_transfer
    return degradation_ratios


if __name__ == '__main__':
    data_processor = DataProcessor('./data/cnf_update.csv')
    X, Y = data_processor.load_and_process_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    print(X_train[0])

    gb_model = GradientBoostModel()
    if os.path.exists(gb_model.model_transfer_path) and os.path.exists(gb_model.model_bitrate_path):
        gb_model.load_model(gb_model.model_transfer_path, gb_model.model_bitrate_path)
    else:
        gb_model.train(X_train, Y_train)
        gb_model.save_model(gb_model.model_transfer_path, gb_model.model_bitrate_path)
        gb_model.load_model('model_transfer.joblib', 'model_bitrate.joblib')

    Y_pred_transfer, Y_pred_bitrate = gb_model.predict(X_test)
    mse_transfer = mean_squared_error(Y_test[:, 0], Y_pred_transfer)
    mse_bitrate = mean_squared_error(Y_test[:, 1], Y_pred_bitrate)

    predict_plot(gb_model.model_transfer, gb_model.model_bitrate, X_test, Y_test)
    y_compare(Y, gb_model.model_transfer.predict(X))

    y = gb_model.model_transfer.predict(np.array([8.35985610e+06, 3.99755308e+07, 1.77689240e+09, 6.00000000e-01]).reshape(1, -1))
    print(calculate_degradation(y))

