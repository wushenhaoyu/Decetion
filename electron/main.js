const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

function createWindow() {
  // 创建浏览器窗口
  const mainWindow = new BrowserWindow({
    width: 1920,
    height: 1280,
    resizable: false, // 禁止调整窗口大小
    webPreferences: {
      nodeIntegration: true, // 根据需要设置
      contextIsolation: false, // 根据需要设置
    }
  });

  // 加载本地 HTML 文件或远程 URL
  mainWindow.loadURL('http://127.0.0.1:8000/');

  // 打开开发者工具（可选）
  // mainWindow.webContents.openDevTools();
}

let djangoProcess;

function startDjangoServer() {
  // 构建 Django 项目的完整路径
  const djangoPath = path.join(__dirname, '..', 'AIdjango', 'manage.py'); // 使用 .. 表示上一级目录

  // 启动 Django 开发服务器
  djangoProcess = spawn('python', [djangoPath, 'runserver', '8080'], {
    cwd: path.join(__dirname, '..'), // 设置工作目录为 AIdjango
    shell: true
  });

  djangoProcess.stdout.on('data', (data) => {
    console.log(`stdout: ${data}`);
  });

  djangoProcess.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  djangoProcess.on('close', (code) => {
    console.log(`Django server exited with code ${code}`);
  });
}

// 当 Electron 完成初始化并准备创建浏览器窗口时，将调用此方法
app.whenReady().then(() => {
  startDjangoServer(); // 启动 Django 服务器
  createWindow(); // 创建 Electron 窗口
});

// 当所有窗口关闭时退出应用
app.on('window-all-closed', function () {
  // 在 macOS 上，除非用户使用 Cmd + Q 明确退出，否则应用会保持活动状态
  if (process.platform !== 'darwin') {
    app.quit();
    if (djangoProcess) {
      djangoProcess.kill(); // 关闭 Django 服务器
    }
  }
});

app.on('activate', function () {
  // 在 macOS 上，当 dock 图标被点击且没有其他窗口打开时，通常会在应用中重新创建一个窗口
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});