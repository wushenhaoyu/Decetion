@echo off
REM 设置当前目录为 BAT 文件所在的目录
cd /d "%~dp0"

REM 获取传入的 Conda 环境名称
set CONDA_ENV=%1

REM 检查是否传入了环境名称
if "%CONDA_ENV%"=="" (
    echo 请提供 Conda 环境名称!
    exit /b
)

REM 激活传入的 Conda 环境
call conda activate %CONDA_ENV%

REM 运行 Django 服务器
python AIdjango/manage.py runserver
