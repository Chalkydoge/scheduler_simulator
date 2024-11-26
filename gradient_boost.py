import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def data_process():
    # 1. 数据加载
    # 假设 CSV 数据文件路径为 "data.csv"
    data = pd.read_csv('./data/cnf_update.csv')
    print(data.head())

    # 过滤掉 replica_num == 0 的数据
    filtered_data = data[data['replica_num'] != 0]

    # 检查 transfer 列的描述性统计
    print(filtered_data['transfer'].describe())

    # 2. 数据预处理
    # 假设 X = [miss, cache_ref, instruction, ipc]， Y = [transfer, bitrate]
    X = data[['miss', 'cache_ref', 'instruction', 'ipc']]
    Y = data[['transfer', 'bitrate']]

    # 处理缺失值
    imputer = SimpleImputer(strategy='mean')  # 使用均值填补缺失值
    X_imputed = imputer.fit_transform(X)
    Y_imputed = imputer.fit_transform(Y)

    # 如果需要对数据进行标准化，可以使用 StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, Y_imputed


def gb_model():
    X_scaled, Y_imputed = data_process()

    # 3. 数据集划分：80%训练集，20%测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_imputed, test_size=0.2, random_state=42)

    # 训练模型：针对 transfer 目标
    model_transfer = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model_transfer.fit(X_train, Y_train[:, 0])  # 只使用 transfer 列作为目标

    # 训练模型：针对 bitrate 目标
    model_bitrate = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model_bitrate.fit(X_train, Y_train[:, 1])  # 只使用 bitrate 列作为目标

    # 预测
    Y_pred_transfer = model_transfer.predict(X_test)
    Y_pred_bitrate = model_bitrate.predict(X_test)

    # 评估模型：均方误差
    mse_transfer = mean_squared_error(Y_test[:, 0], Y_pred_transfer)
    mse_bitrate = mean_squared_error(Y_test[:, 1], Y_pred_bitrate)
    print(f'MSE for transfer: {mse_transfer}')
    print(f'MSE for bitrate: {mse_bitrate}')

    return model_transfer, model_bitrate


def predict_plot(model_transfer, model_bitrate, x, y):
    # 预测
    Y_pred_transfer = model_transfer.predict(x)
    Y_pred_bitrate = model_bitrate.predict(x)

    # 可视化：转移率 (transfer) 对比预测值和实际值
    plt.figure(figsize=(10, 6))

    # 转移率对比
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y[:, 0], y=Y_pred_transfer, color='blue')
    plt.plot([y[:, 0].min(), y[:, 0].max()], [y[:, 0].min(), y[:, 0].max()], 'r--', lw=2)
    plt.title('Transfer Rate: Predicted vs Actual')
    plt.xlabel('Actual Transfer Rate')
    plt.ylabel('Predicted Transfer Rate')

    # 比特率 (bitrate) 对比
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y[:, 1], y=Y_pred_bitrate, color='green')
    plt.plot([y[:, 1].min(), y[:, 1].max()], [y[:, 1].min(), y[:, 1].max()], 'r--', lw=2)
    plt.title('Bitrate: Predicted vs Actual')
    plt.xlabel('Actual Bitrate')
    plt.ylabel('Predicted Bitrate')

    plt.tight_layout()
    plt.savefig('sample_model1.png')


def y_compare(ytest, ypred):
    data = pd.read_csv('./data/cnf_update.csv')
    # 假设 replica_num 是数据中的一个列，可以直接取
    # 如果没有这个列，可以通过 `range(len(Y_test))` 来模拟
    replica_num = data['replica_num']  # 假设数据中有这列

    # 跳过 replica_num=0 的数据点
    mask = replica_num != 0  # 过滤掉 replica_num=0 的数据
    filtered_replica_num = replica_num[mask]
    filtered_actual_transfer = ytest[mask, 0]  # 实际的 transfer
    filtered_predicted_transfer = ypred[mask]  # 预测的 transfer

    # 创建一个大的图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制主图：x 轴为 replica_num，y 轴为 transfer
    ax.plot(filtered_replica_num, filtered_actual_transfer, label='Actual Transfer', color='blue', marker='o',
            linestyle='-', markersize=4)
    ax.plot(filtered_replica_num, filtered_predicted_transfer, label='Predicted Transfer', color='red', linestyle='--',
            marker='x', markersize=4)

    # 添加标题和标签
    ax.set_title('Transfer Rate: Actual vs Predicted by Replica Number (Excluding replica_num=0)')
    ax.set_xlabel('Replica Number')
    ax.set_ylabel('Transfer Rate')

    # 显示图例
    ax.legend()

    # 放大 0-20 区域
    axins = inset_axes(ax, width="30%", height="30%", loc='upper left', borderpad=2)  # 创建一个放大区域

    # 在放大区域绘制相同的内容，只选取 0 到 20 之间的数据
    axins.plot(filtered_replica_num, filtered_actual_transfer, label='Actual Transfer', color='blue', marker='o',
               linestyle='-', markersize=4)
    axins.plot(filtered_replica_num, filtered_predicted_transfer, label='Predicted Transfer', color='red',
               linestyle='--', marker='x', markersize=4)

    # 设置放大区域的 x 和 y 轴范围
    axins.set_xlim(0, 20)
    axins.set_ylim(min(filtered_actual_transfer[filtered_replica_num <= 20].min(),
                       filtered_predicted_transfer[filtered_replica_num <= 20].min()) - 0.1,
                   max(filtered_actual_transfer[filtered_replica_num <= 20].max(),
                       filtered_predicted_transfer[filtered_replica_num <= 20].max()) + 0.1)

    # 添加放大区域的边框
    axins.set_xticks([0, 5, 10, 15, 20])
    axins.set_yticks([min(filtered_actual_transfer[filtered_replica_num <= 20].min(),
                          filtered_predicted_transfer[filtered_replica_num <= 20].min()) - 0.1,
                      (max(filtered_actual_transfer[filtered_replica_num <= 20].max(),
                           filtered_predicted_transfer[filtered_replica_num <= 20].max()) +
                       min(filtered_actual_transfer[filtered_replica_num <= 20].min(),
                           filtered_predicted_transfer[filtered_replica_num <= 20].min())) / 2,
                      max(filtered_actual_transfer[filtered_replica_num <= 20].max(),
                          filtered_predicted_transfer[filtered_replica_num <= 20].max()) + 0.1])

    # 在主图中绘制放大区域的框架
    ax.indicate_inset_zoom(axins)

    # 保存图像到文件，dpi=300
    plt.savefig('transfer_rate_comparison_zoomed_no_0.png', dpi=300)

    # 显示图像
    plt.show()


if __name__ == '__main__':
    x, y = data_process()
    m1, m2 = gb_model()
    predict_plot(m1, m2, x, y)
    y_compare(ytest=y, ypred=m1.predict(x))