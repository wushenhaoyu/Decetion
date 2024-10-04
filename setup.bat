@echo off
REM Step 0: 设置代码页为 UTF-8，防止中文乱码
chcp 65001 >nul

REM Step 1: 检查是否安装了Python
echo 检查是否已安装 Python...

python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python 未安装，请先安装 Python.
    exit /b 1
)

REM Step 2: 检查虚拟环境是否已创建
IF NOT EXIST "venv" (
    echo 创建虚拟环境...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo 虚拟环境创建失败，请检查 Python 安装。
        exit /b 1
    )
) ELSE (
    echo 虚拟环境已存在，跳过创建。
)

REM Step 3: 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM Step 4: 升级 pip
echo 升级 pip...
python -m pip install --upgrade pip

REM Step 5: 安装 requirements.txt 中的依赖
IF EXIST "requirements.txt" (
    echo 安装 requirements.txt 中的依赖...
    pip install -r requirements.txt
    IF ERRORLEVEL 1 (
        echo 依赖安装失败，请检查 requirements.txt 文件。
        deactivate
        pause
        exit /b 1
    )
) ELSE (
    echo 没有找到 requirements.txt 文件。
)

echo 完成！虚拟环境已激活并安装了所有依赖。

pause
