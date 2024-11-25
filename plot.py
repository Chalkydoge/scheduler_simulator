import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def load_csv():
    # 读取 CSV 文件
    df = pd.read_csv('./data/cnf_update.csv')  # 替换为你的 CSV 文件路径
    # 检查数据是否正确加载
    print(df.head())
    return df


def draw_scatter():
    load_csv()
    data = {
        'cache_miss': np.random.randint(1e9, 1e10, 100),
        'cache_reference': np.random.randint(1e10, 1e11, 100),
        'ipc': np.random.uniform(0.77, 3.2, 100),
        'transfer_performance': np.random.randint(1000, 2000, 100),
        'bitrate_performance': np.random.randint(500, 800, 100)
    }

    df = pd.DataFrame(data)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制 cache_miss 和 cache_reference 与 transfer_performance 和 bitrate_performance 的散点图
    # 使用不同颜色表示 transfer_performance 和 bitrate_performance
    ax.scatter(df['cache_miss'], df['transfer_performance'], color='blue', label='Transfer Performance (cache_miss)',
               alpha=0.6)
    ax.scatter(df['cache_miss'], df['bitrate_performance'], color='green', label='Bitrate Performance (cache_miss)',
               alpha=0.6)

    ax.scatter(df['cache_reference'], df['transfer_performance'], color='blue',
               label='Transfer Performance (cache_reference)', alpha=0.6)
    ax.scatter(df['cache_reference'], df['bitrate_performance'], color='green',
               label='Bitrate Performance (cache_reference)', alpha=0.6)

    ax.scatter(df['ipc'], df['transfer_performance'], color='blue', label='Transfer Performance (IPC)', alpha=0.6)
    ax.scatter(df['ipc'], df['bitrate_performance'], color='green', label='Bitrate Performance (IPC)', alpha=0.6)

    # 设置标签和标题
    ax.set_xlabel('Cache Miss / Cache Reference / IPC')
    ax.set_ylabel('Transfer Performance / Bitrate Performance')
    plt.title('Scatter Plot of Performance Metrics')

    # 添加图例
    ax.legend()

    # 保存图像到文件
    plt.savefig('performance_scatter_plot.png', dpi=300)

    # 显示图形（如果需要）
    plt.show()

    # 如果需要关闭图像，避免内存占用
    plt.close()


def draw_doubley():
    # 模拟数据
    data = {
        'cache_miss': np.random.randint(1e9, 1e10, 100),
        'cache_reference': np.random.randint(1e10, 1e11, 100),
        'ipc': np.random.uniform(0.77, 3.2, 100),
        'transfer_performance': np.random.randint(1000, 2000, 100),
        'bitrate_performance': np.random.randint(500, 800, 100)
    }

    df = pd.DataFrame(data)

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 cache_miss 和 cache_reference（左Y轴）
    ax1.set_xlabel('index')
    ax1.set_ylabel('Cache Miss / Cache Reference', color='tab:red')
    ax1.plot(df.index, df['cache_miss'], color='tab:red', label='Cache Miss')
    ax1.plot(df.index, df['cache_reference'], color='tab:orange', label='Cache Reference')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 创建第二个 Y 轴，共享 X 轴（右Y轴）
    ax2 = ax1.twinx()
    ax2.set_ylabel('IPC / Transfer Performance / Bitrate Performance', color='tab:blue')
    ax2.plot(df.index, df['ipc'], color='tab:blue', label='IPC')
    ax2.plot(df.index, df['transfer_performance'], color='tab:green', label='Transfer Performance')
    ax2.plot(df.index, df['bitrate_performance'], color='tab:purple', label='Bitrate Performance')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 添加图例
    fig.tight_layout()  # 调整布局避免重叠
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # 设置标题
    plt.title('Multiple metrics')

    # 保存图像到文件
    plt.savefig('performance_comparison.png', dpi=300)  # 保存为 PNG 文件，可以更改为其他格式如 .jpg, .pdf 等

    # 如果需要关闭图像，避免内存占用
    plt.close()


def draw_multiple_scatter():
    # 读取 CSV 文件
    df = pd.read_csv('./data/cnf_update.csv')  # 替换为你的 CSV 文件路径

    # 检查数据是否正确加载
    print(df.head())

    # 创建一个 2x3 的子图网格，2 行 3 列
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    df = df.sort_values(by='transfer_performance')
    # 绘制每个子图：每个子图的 x 轴为 cache_miss, cache_reference, ipc，y 轴为 transfer_performance 和 bitrate_performance
    axes[0, 0].scatter(df['cache_miss'], df['transfer_performance'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Cache Miss')
    axes[0, 0].set_ylabel('Transfer Performance')
    axes[0, 0].set_title('Cache Miss vs Transfer Performance')

    axes[0, 1].scatter(df['cache_reference'], df['transfer_performance'], color='blue', alpha=0.6)
    axes[0, 1].set_xlabel('Cache Reference')
    axes[0, 1].set_ylabel('Transfer Performance')
    axes[0, 1].set_title('Cache Reference vs Transfer Performance')

    axes[0, 2].scatter(df['ipc'], df['transfer_performance'], color='blue', alpha=0.6)
    axes[0, 2].set_xlabel('IPC')
    axes[0, 2].set_ylabel('Transfer Performance')
    axes[0, 2].set_title('IPC vs Transfer Performance')

    df = df.sort_values(by='bitrate_performance')
    axes[1, 0].scatter(df['cache_miss'], df['bitrate_performance'], color='green', alpha=0.6)
    axes[1, 0].set_xlabel('Cache Miss')
    axes[1, 0].set_ylabel('Bitrate Performance')
    axes[1, 0].set_title('Cache Miss vs Bitrate Performance')

    axes[1, 1].scatter(df['cache_reference'], df['bitrate_performance'], color='green', alpha=0.6)
    axes[1, 1].set_xlabel('Cache Reference')
    axes[1, 1].set_ylabel('Bitrate Performance')
    axes[1, 1].set_title('Cache Reference vs Bitrate Performance')

    axes[1, 2].scatter(df['ipc'], df['bitrate_performance'], color='green', alpha=0.6)
    axes[1, 2].set_xlabel('IPC')
    axes[1, 2].set_ylabel('Bitrate Performance')
    axes[1, 2].set_title('IPC vs Bitrate Performance')

    # 调整子图布局，避免重叠
    fig.tight_layout()

    # 保存图像到文件
    plt.savefig('performance_scatter_plots.png', dpi=300)

    # 显示图形（如果需要）
    plt.show()

    # 如果需要关闭图像，避免内存占用
    plt.close()


def draw_new_scatter():
    # 读取CSV文件
    data = load_csv()
    data['ipc'] = data['ipc'].astype(float)
    data['ipc'] = data['ipc'].apply(lambda x: f"{x:.4f}")
    data['ipc'] = data['ipc'].astype(float)

    print(data['ipc'])
    # 根据 replica_num 排序数据
    data = data.sort_values(by='replica_num')

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 创建一个子图网格
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 创建一个子图网格
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 定义 x 轴刻度
    x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # 绘制 miss vs replica_num 散点图
    sns.scatterplot(x='replica_num', y='miss', data=data, ax=axes[0, 0])
    axes[0, 0].set_title('Miss vs Replica Num')
    axes[0, 0].set_xlabel('Replica Num')
    axes[0, 0].set_ylabel('Miss')
    axes[0, 0].set_xticks(x_ticks)
    axes[0, 0].set_xticklabels(x_ticks)
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 绘制 cache_ref vs replica_num 散点图
    sns.scatterplot(x='replica_num', y='cache_ref', data=data, ax=axes[0, 1])
    axes[0, 1].set_title('Cache Ref vs Replica Num')
    axes[0, 1].set_xlabel('Replica Num')
    axes[0, 1].set_ylabel('Cache Ref')
    axes[0, 1].set_xticks(x_ticks)
    axes[0, 1].set_xticklabels(x_ticks)
    axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 绘制 instruction vs replica_num 散点图
    sns.scatterplot(x='replica_num', y='instruction', data=data, ax=axes[1, 0])
    axes[1, 0].set_title('Instruction vs Replica Num')
    axes[1, 0].set_xlabel('Replica Num')
    axes[1, 0].set_ylabel('Instruction')
    axes[1, 0].set_xticks(x_ticks)
    axes[1, 0].set_xticklabels(x_ticks)
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 绘制 ipc vs replica_num 散点图
    sns.scatterplot(x='replica_num', y='ipc', data=data, ax=axes[1, 1])
    axes[1, 1].set_title('IPC vs Replica Num')
    axes[1, 1].set_xlabel('Replica Num')
    axes[1, 1].set_ylabel('IPC')
    axes[1, 1].set_xticks(x_ticks)
    axes[1, 1].set_xticklabels(x_ticks)
    axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    # 调整布局
    plt.tight_layout()

    # 保存图表到PNG文件
    plt.savefig('scatter_plots.png')


if __name__ == '__main__':
    draw_new_scatter()