import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. 数据加载
# 假设 CSV 数据文件路径为 "data.csv"
data = pd.read_csv('data.csv')

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

# 3. 数据集划分：80%训练集，20%测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_imputed, test_size=0.2, random_state=42)

# 4. 初始化 Gradient Boosting 回归模型
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 5. 训练模型
model.fit(X_train, Y_train)

# 6. 模型预测
Y_pred = model.predict(X_test)

# 7. 评估模型：计算均方误差（MSE）
mse = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error: {mse}')

# 如果需要，可以对每个目标变量单独进行评估：
mse_transfer = mean_squared_error(Y_test[:, 0], Y_pred[:, 0])
mse_bitrate = mean_squared_error(Y_test[:, 1], Y_pred[:, 1])

print(f'MSE for transfer: {mse_transfer}')
print(f'MSE for bitrate: {mse_bitrate}')

# 8. 模型的可解释性：查看重要特征
feature_importances = model.feature_importances_
print('Feature Importances:', feature_importances)
