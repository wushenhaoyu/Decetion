const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let win1;
let win2;

function createWindow() {
    // 创建主窗口
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
    
    // 创建第二个窗口（如有必要）
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
            contextIsolation: true, // 启用上下文隔离
        }
    });
    win2.hide()
}
ipcMain.on('enter-page',()=>{
    win2.loadURL(('http://localhost:8080'));
    win2.show()
    //win1.hide()
})
ipcMain.handle('bat-run', async () => {
    const batFilePath = path.join(__dirname, '../django.bat'); // 替换为你的 BAT 文件路径
    console.log('Running BAT file:', batFilePath);
    
    // 启动 Django 服务器
    const batProcess = spawn('cmd.exe', ['/c', batFilePath]);

    // 监听输出数据并发送到前端
    batProcess.stdout.on('data', (data) => {
        const output = data.toString();
        win1.webContents.send('bat-output', output);

        // 根据输出内容判断各个模块是否成功初始化
        if (output.includes('Haze Remover')) {
            win1.webContents.send('bat-status',1);
        } else if (output.includes('Video Enhancer')) {
            win1.webContents.send('bat-status',1);
        } else if (output.includes('Model Configuration')) {
            win1.webContents.send('bat-status',1);
        } else if (output.includes('Traceback')) {
            win1.webContents.send('bat-status',2);
        } else if (output.includes('CTRL-BREAK')) {
            win1.webContents.send('bat-status',3);
        }
    });

    // 监听错误输出并发送到前端
    batProcess.stderr.on('data', (data) => {
        win1.webContents.send('bat-output', `Error: ${data.toString()}`);
    });

    // 监听进程关闭
    batProcess.on('close', (code) => {
        win1.webContents.send('bat-output', `Process exited with code: ${code}`);
    });

    return { success: true };
});

// 当应用准备好时创建窗口
app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

// 所有窗口关闭时退出应用
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});
