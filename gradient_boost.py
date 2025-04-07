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

"""
[NF], EXEC,IPC,FREQ,AFREQ,L3MISS,L2MISS,L3HIT,L2HIT,L3MPI,L2MPI,L3OCC,TEMP,READ,WRITE,LLCRDMISSLAT,INST,TRANSFER,BITRATE
nginx, 0.01,0.53,0.02,0.57,832000,3159000,0.74,0.36,0.00,0.01,14352,0.07,0.03,145.15,343000000
"""
FIELD_NAMES = [
    'EXEC',
    'IPC',
    'FREQ',
    'AFREQ',
    'L3MISS',
    'L2MISS',
    'L3HIT',
    'L2HIT',
    'L3MPI',
    'L2MPI',
    'L3OCC',
    'READ',
    'WRITE',
    'LLCRDMISSLAT',
    'INST',
]

FIELD_INDEX = {
    'EXEC': 0,
    'IPC': 1,
    'FREQ': 2,
    'AFREQ': 3,
    'L3MISS': 4,
    'L2MISS': 5,
    'L3HIT': 6,
    'L2HIT': 7,
    'L3MPI': 8,
    'L2MPI': 9,
    'L3OCC': 10,
    'READ': 11,
    'WRITE': 12,
    'LLCRDMISSLAT': 13,
    'INST': 14,
}

MODEL_TRANSFER_CKPT = './ckpt/updated_model_transfer.joblib'
NF_MAP = {
    'nginx': 0,
    'pktstat': 1,
    'snort': 2,
    'squid': 3,
    'ufw': 4,
    'iperf': 5,
}

# Gbps, baseline transfer rate
BASELINE_TRANSFER_MAP = {
    'nginx': 6.85,
    'pktstat': 25.0,
    'snort': 16.9,
    'squid': 25.2,
    'ufw': 11.1,
    'iperf': 25.6,
}

foo = {
    'nginx': 1,
    'snort': 1,
    'pktstat': 1,
}


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
        X = data[['NF', 'EXEC','IPC','FREQ','AFREQ','L3MISS','L2MISS','L3HIT','L2HIT','L3MPI','L2MPI','L3OCC','READ','WRITE','LLCRDMISSLAT','INST']]
        Y = data[['TRANSFER']]

        # 处理缺失值
        _imp = SimpleImputer(strategy='mean')
        x_imputed = _imp.fit_transform(X)
        y_imputed = _imp.fit_transform(Y)

        # 标准化 去掉因为后面还需要推理 输入的x是原始数据
        return x_imputed, y_imputed


class GradientBoostModel:
    def __init__(self):
        self.model_transfer = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3,
                                                        random_state=42)
        self.model_transfer_path = MODEL_TRANSFER_CKPT

    def train(self, x_train, y_train):
        # y_train的第一个维度对应于传输率
        self.model_transfer.fit(x_train, y_train[:, 0])

    def save_model(self, filepath_transfer):
        if not os.path.exists(os.path.dirname(filepath_transfer)):
            os.makedirs(os.path.dirname(filepath_transfer))
        joblib.dump(self.model_transfer, filepath_transfer)

    def load_model(self, filepath_transfer):
        self.model_transfer = joblib.load(filepath_transfer)

    def predict(self, x):
        y_pred_transfer = self.model_transfer.predict(x)
        return y_pred_transfer


def predict_plot(model_transfer, X, Y):
    Y_pred_transfer = model_transfer.predict(X)

    plt.figure(figsize=(14, 6))

    # 绘制 Transfer Rate: Predicted vs Actual
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=Y[:, 0], y=Y_pred_transfer, color='blue', label='Predicted Transfer Rate')
    sns.scatterplot(x=Y[:, 0], y=Y[:, 0], color='red', label='Actual Transfer Rate')
    plt.plot([Y[:, 0].min(), Y[:, 0].max()], [Y[:, 0].min(), Y[:, 0].max()], 'r--', lw=2)
    plt.title('Transfer Rate: Predicted vs Actual')
    plt.xlabel('Actual Transfer Rate')
    plt.ylabel('Predicted Transfer Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig('throughput_predict.png')


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


def calculate_degradation(ypred, target_nf):
    # 计算每个 replica_num 的 transfer 相对于基准值的比值
    baseline_transfer = BASELINE_TRANSFER_MAP[target_nf]
    degradation_ratios = ypred / baseline_transfer
    return degradation_ratios



def synthesize(already_scheduled_nf: dict[str, int]):
    # 根据已经调度的nf，预测下一个nf的transfer rate
    base_vectors = {
        'nginx': [0.01,0.76,0.01,0.57,619000,1761000,0.65,0.35,0.0,0.01,14976.0,0.06,0.02,213.28,269000000,],
        'snort': [0.01,0.78,0.01,0.6,556000,1538000,0.64,0.35,0.0,0.01,14184.0,0.05,0.02,227.91,251000000,],
        'pktstat': [0.01,0.67,0.02,0.62,673000,2243000,0.7,0.36,0.0,0.01,14136.0,0.06,0.02,210.64,317000000,],
        'squid': [0.01,0.76,0.01,0.62,477000,1435000,0.67,0.37,0.0,0.01,15288.0,0.04,0.02,243.35,232000000,],
        'ufw': [0.01,0.81,0.02,0.62,728000,2094000,0.65,0.42,0.0,0.01,14520.0,0.06,0.03,219.3,368000000],
    }

    # 根据已经调度的nf，预测下一个nf的transfer rate
    m = len(base_vectors[list(already_scheduled_nf.keys())[0]])
    base_vector = [0] * m
    for nf in already_scheduled_nf:
        for field in FIELD_NAMES:
            if field in ["L3MISS", "L2MISS", "L3MPI", "L2MPI", "L3OCC", "INST", "READ", "WRITE"]:
                base_vector[FIELD_INDEX[field]] += base_vectors[nf][FIELD_INDEX[field]]
            elif field in ["EXEC", "IPC", "FREQ", "AFREQ"]:
                base_vector[FIELD_INDEX[field]] += base_vectors[nf][FIELD_INDEX[field]]
            elif field in ["L3HIT", "L2HIT"]:
                ratio = (0.62 / 0.65, 0.32 / 0.35)
                base_vector[FIELD_INDEX[field]] += base_vectors[nf][FIELD_INDEX[field]] * ratio[FIELD_INDEX[field]]

    print(base_vector)
    return base_vector


if __name__ == '__main__':
    """
        当前的cnf_update2.csv的targetNF都是nginx
        需要将targetNF替换为NF_MAP中的值
    """
    data_processor = DataProcessor('./data/cnf_update3.csv')
    X, Y = data_processor.load_and_process_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

    gb_model = GradientBoostModel()
    if os.path.exists(gb_model.model_transfer_path):
        gb_model.load_model(gb_model.model_transfer_path)
    else:
        gb_model.train(X_train, Y_train)
        gb_model.save_model(gb_model.model_transfer_path)
        gb_model.load_model(MODEL_TRANSFER_CKPT)

    Y_pred_transfer = gb_model.predict(X_test)
    mse_transfer = mean_squared_error(Y_test[:, 0], Y_pred_transfer)

    predict_plot(gb_model.model_transfer, X_test, Y_test)

    y = gb_model.model_transfer.predict(np.array([NF_MAP['nginx'],0.01,0.53,0.02,0.57,832000,3159000,0.74,0.36,0.00,0.01,14352,0.07,0.03,145.15,343000000]).reshape(1, -1))
    print(calculate_degradation(y, 'nginx'))

    y = gb_model.model_transfer.predict(np.array([NF_MAP['snort'], 0.01,0.55,0.01,0.50,552000,2001000,0.72,0.28,0.00,0.01,15528,0.05,0.03,153.53,222000000]).reshape(1, -1))
    print(calculate_degradation(y, 'snort'))
