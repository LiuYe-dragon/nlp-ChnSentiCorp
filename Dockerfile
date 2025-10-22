# 使用轻量级 Python 镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 加载依赖文件
COPY requirements.txt /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 避免 TensorFlow 提示警告
ENV TF_CPP_MIN_LOG_LEVEL=2

# 复制项目文件
COPY . /app

# 默认使用 UTF-8 编码
ENV PYTHONUNBUFFERED=1

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动 FastAPI 服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]