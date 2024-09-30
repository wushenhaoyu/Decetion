@echo off
REM 设置当前目录为 BAT 文件所在的目录
cd /d "%~dp0"

REM 激活 Anaconda 环境
call conda activate paddle

REM 运行 Django 服务器
python AIdjango/manage.py runserver 

