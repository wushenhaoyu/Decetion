#!/bin/bash

# 获取传入的 Conda 环境名称
CONDA_ENV=$1

# 检查是否传入了环境名称
if [ -z "$CONDA_ENV" ]; then
    echo "请提供 Conda 环境名称!"
    exit 1
fi

conda activate "$CONDA_ENV"

# 运行 Django 服务器
python AIdjango/manage.py runserver
