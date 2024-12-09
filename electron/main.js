const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

let mainWindow;
let djangoProcess;
let condaEnvironments = []; // 存储 Anaconda 环境值

// 创建窗口
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1100,
        height: 640,
        resizable: false, // 禁止调整窗口大小
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'), // 配置 preload.js
            nodeIntegration: false, // 不允许直接使用 Node.js
            contextIsolation: true, // 确保隔离上下文
        },
    });

    // 加载页面
    mainWindow.loadFile('SelectInspiration/index6.html');
}

// 获取 Conda 环境列表
function listAnacondaEnvironments() {
    return new Promise((resolve, reject) => {
        const condaProcess = spawn('conda', ['env', 'list']);

        let output = '';

        condaProcess.stdout.on('data', (data) => {
            output += data.toString();
        });

        condaProcess.stderr.on('data', (data) => {
            console.error(`Error listing conda environments:\n${data}`);
        });

        condaProcess.on('close', (code) => {
            if (code === 0) {
                // 解析 Conda 环境列表
                const lines = output.split('\n');
                condaEnvironments = lines.slice(2).map(line => line.trim().split(/\s+/)[0]).filter(env => env);
                console.log(`Conda environments: ${condaEnvironments.join(', ')}`);
                resolve(condaEnvironments); // 返回环境列表
            } else {
                reject('Failed to list conda environments');
            }
        });
    });
}

// 启动 Django 服务器
const net = require('net'); // 引入 net 模块用于端口检测

function checkPort(port) {
    return new Promise((resolve) => {
        const server = net.createServer();

        server.once('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                resolve(true); // 端口被占用
            } else {
                resolve(false); // 其他错误
            }
        });

        server.once('listening', () => {
            server.close();
            resolve(false); // 端口未被占用
        });

        server.listen(port);
    });
}
const http = require('http');

function startDjangoServer(condaEnv) {
    // 检查端口是否被占用
    checkPortAvailability('http://127.0.0.1:8000/')
        .then((isAvailable) => {
            if (!isAvailable) {
                console.log('Port 8000 is already in use. Skipping Django startup.');
                mainWindow.setSize(1440, 960);
                mainWindow.setTitle('路况瞭望');
                mainWindow.loadURL('http://127.0.0.1:8000/'); // 直接加载已有的 Django 服务
                return;
            }

            console.log('Port 8000 is available. Starting Django server...');
            let scriptPath;

            if (process.platform === 'win32') {
                // Windows 使用 .bat 文件
                scriptPath = path.join(__dirname, 'django.bat');
                console.log(path.join(__dirname));
                djangoProcess = spawn('cmd', ['/c', scriptPath, condaEnv], {
                    cwd: path.join(__dirname), // 设置工作目录为项目根目录
                    shell: true // 使用 shell 执行命令
                });
            } else if (process.platform === 'linux') {
                // Linux 使用 .sh 脚本
                scriptPath = path.join(__dirname, 'django.sh');
                console.log(path.join(__dirname));
                djangoProcess = spawn('bash', [scriptPath, condaEnv], {
                    cwd: path.join(__dirname), // 设置工作目录为项目根目录
                    shell: true // 使用 shell 执行命令
                });
            } else {
                console.error('Unsupported platform');
                return;
            }

            // 处理 Django 进程的标准输出
            djangoProcess.stdout.on('data', (data) => {
                const output = data.toString();
                console.log(`stdout: ${output}`);

                // 检查 Django 是否启动成功
                if (output.includes('Performing system checks...')) {
                    mainWindow.loadFile('index.html');
                }

                if (output.includes("CTRL-BREAK")) {
                    console.log('Quit');
                    mainWindow.setSize(1440, 960);
                    mainWindow.setTitle('路况瞭望');
                    mainWindow.loadURL('http://127.0.0.1:8000/'); // 成功启动 Django 后加载网页
                }
            });

            // 处理 Django 进程的标准错误
            djangoProcess.stderr.on('data', (data) => {
                console.error(`stderr: ${data}`);
                if (data.toString().includes("Error")) {
                    console.error('Failed to start Django server. Closing application.');
                    if (mainWindow) {
                        mainWindow.close();
                    }
                    app.quit();
                }
            });

            // 处理 Django 进程的退出
            djangoProcess.on('close', (code) => {
                console.log(`Django server exited with code ${code}`);
            });
        })
        .catch((err) => {
            console.error('Error checking port availability:', err);
        });
}

// 检查端口是否可用
function checkPortAvailability(url) {
    return new Promise((resolve) => {
        const req = http.get(url, () => {
            console.log(`Port is in use: ${url}`);
            resolve(false); // 如果请求成功，说明端口被占用
        });

        req.on('error', () => {
            console.log(`Port is available: ${url}`);
            resolve(true); // 如果请求失败，说明端口可用
        });

        req.end();
    });
}



// 监听从渲染进程请求 Conda 环境列表的事件
ipcMain.handle('get-conda-environments', async () => {
    try {
        if (condaEnvironments.length === 0) {
            // 如果 Conda 环境列表为空，则重新查询
            await listAnacondaEnvironments();
        }
        return condaEnvironments; // 返回当前存储的 Conda 环境列表
    } catch (error) {
        console.error('Error getting Conda environments:', error);
        return []; // 如果发生错误，返回空数组
    }
});

// 监听从渲染进程发送的环境选择事件
ipcMain.on('selected-environment', (event, condaEnv) => {
    console.log(`Selected environment: ${condaEnv}`);
    startDjangoServer(condaEnv); // 使用选中的环境启动 Django 服务器
});

// 初始化 Electron 应用
app.whenReady().then(() => {
    listAnacondaEnvironments() // 查询 Conda 环境
        .then(() => createWindow()) // 成功查询环境后创建窗口
        .catch((error) => {
            console.error('Error during app initialization:', error);
            app.quit(); // 如果查询失败，退出应用
        });
});

// 确保退出时杀死 Django 进程
function terminateDjangoProcess() {
    if (djangoProcess) {
        if (process.platform === 'win32') {
            // Windows 下发送 Ctrl+C 信号
            const readline = require('readline');
            readline.createInterface({
                input: process.stdin,
                output: process.stdout,
            }).on('SIGINT', () => {
                // 模拟 Ctrl+C
                process.emit('SIGINT');
            });
        } else {
            // 其他平台发送 SIGINT 信号
            djangoProcess.kill('SIGINT');
        }

        djangoProcess = null; // 避免重复调用
    }
}

// 在 `window-all-closed` 和 `before-quit` 中调用
app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') {
        terminateDjangoProcess();
        app.quit(); // 确保退出应用
    }
});

app.on('before-quit', () => {
    terminateDjangoProcess();
});

// 在 macOS 上，重新激活窗口
app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
