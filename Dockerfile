# 设置基础镜像
FROM continuumio/miniconda3:latest

# 复制环境配置文件到工作目录
COPY environment.yml /app/environment.yml

# 创建conda环境
RUN conda env create -f /app/environment.yml

# 激活conda环境
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# 设置工作目录
WORKDIR /app

# 复制应用程序文件到工作目录
COPY . /app

# 运行应用程序
CMD ["python", "app.py"]