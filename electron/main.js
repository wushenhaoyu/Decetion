const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let win1;
let win2;

function createWindow() {
    win1 = new BrowserWindow({
        width: 800,
        height: 600,
        autoHideMenuBar: true,
        alwaysOnTop: true,
        x: 0,
        y: 0,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false, // 推荐关闭 nodeIntegration
            contextIsolation: true, // 启用上下文隔离
        }
    });

    win1.loadFile('./pages/index/index.html');
    
    win2 = new BrowserWindow({
        width: 960,
        height: 540,
        autoHideMenuBar: true,
        alwaysOnTop: true,
        x: 800,
        y: 0,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false, // 推荐关闭 nodeIntegration
            contextIsolation: true,// 启用上下文隔离
        }
    })
    
}

ipcMain.handle('bat-run', async () => {
    const batFilePath = path.join(__dirname, '../django.bat'); // 替换为你的 BAT 文件路径
    console.log(batFilePath);
    const batProcess = spawn('cmd.exe', ['/c', batFilePath]);

    batProcess.stdout.on('data', (data) => {
        // 将输出发送到前端
        win1.webContents.send('bat-output', data.toString());
    });

    batProcess.stderr.on('data', (data) => {
        // 处理错误输出
        win1.webContents.send('bat-output', `Error: ${data.toString()}`);
    });

    batProcess.on('close', (code) => {
        win1.webContents.send('bat-output', `Process exited with code: ${code}`);
    });

    return { success: true };
});

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});
