import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

sample_data = [
    (1, 6.526),
    (2, 5.756),
    (3, 5.118),
    (4, 4.422),
    (5, 3.942),
    (6, 3.612),
    (7, 3.198),
    (8, 2.717),
    (9, 2.433),
    (10, 2.070),
    (16, 1.50),
    (22, 1.32),
    (23, 1.31),
    (25, 1.27),
    (30, 1.307),
    (50, 1.307),
    (60, 1.36),
    (75, 1.31),
    (100, 1.252)
]

x = np.array([item[0] for item in sample_data]).reshape(-1, 1)  # 第一列为 x
y = np.array([((sample_data[0][1] - item[1]) / sample_data[0][1]) * 100 for item in sample_data])  # 计算下降幅度百分比
print(y)


def predict_poly_drop(x_value):
    # 使用多项式回归
    poly = PolynomialFeatures(degree=5)  # 使用二次多项式
    x_poly = poly.fit_transform(x)
    model_poly = LinearRegression()
    model_poly.fit(x_poly, y)
    return model_poly.predict(poly.transform([[x_value]]))[0]


if __name__ == '__main__':
    # 预测示例
    x_test = 15  # 假设我们预测 x=15 时的下降幅度
    predicted_drop = predict_poly_drop(x_test)
    print(f"5次多项式预测 x={x_test} 时的下降幅度为: {predicted_drop:.2f}%")
    print((1.0 - predicted_drop / 100.0) * sample_data[0][1])

    # 可选: 绘制数据点和回归直线
    poly = PolynomialFeatures(degree=5)  # 使用二次多项式
    x_poly = poly.fit_transform(x)
    model_poly = LinearRegression()
    model_poly.fit(x_poly, y)

    # svr model
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)  # 使用径向基函数(RBF)内核
    svr.fit(x, y)
    predicted_drop_svr = svr.predict([[x_test]])[0]
    print(f"SVR预测 x={x_test} 时的下降幅度为: {predicted_drop_svr:.2f}%")
    print((1.0 - predicted_drop_svr / 100.0) * sample_data[0][1])

    x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    y_range_pred = model_poly.predict(x_range_poly)

    # svr
    # y_range_pred = svr.predict(x_range)

    y_range_pred = model_poly.predict(x_range_poly)
    print(y_range_pred)

    plt.scatter(x, y, color='blue', label='data')
    plt.plot(x_range, y_range_pred, color='red', label='Predicted rate')
    plt.xlabel('x=num-of-replicas')
    plt.ylabel('Drop percentage%')
    plt.ylim(0, 100)
    plt.legend()

    plt.gca().invert_yaxis()
    plt.savefig('sample2.png')
