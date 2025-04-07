# 使用官方的 Python 作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 将当前目录下的 requirements.txt 复制到镜像中的 /app 目录
COPY requirements.txt /app/

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 将当前目录下的所有文件复制到镜像中的 /app 目录
COPY . /app/

# 暴露应用运行的端口（假设 Flask 默认在 5000 端口运行）
EXPOSE 5000

# 设置环境变量，禁用 Python 缓存
ENV PYTHONUNBUFFERED=1

# 在启动 Flask 应用之前执行 gradient_boost.py
RUN python3 gradient_boost.py

# 启动 Flask 应用
CMD ["python", "app.py"]
