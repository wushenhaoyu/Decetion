const { contextBridge, ipcRenderer } = require('electron');

console.log('preload');

contextBridge.exposeInMainWorld('electronAPI', {
    action: () => {
        return ipcRenderer.invoke('action'); // 使用 invoke 调用并返回 Promise
    },
    runBatFile: () => {
        return ipcRenderer.invoke('bat-run'); // 调用 bat-run
    },
    onBatOutput: (callback) => {
        ipcRenderer.on('bat-output', (event, data) => {
            callback(data); // 将 bat 输出传递给回调
        });
    },
    onBatStatus: (callback) => {
        ipcRenderer.on('bat-status', (event, data) => {
            callback(data); // 将 bat 状态传递给回调
        });
    },
    enterPage: () => {
       ipcRenderer.send('enter-page');
    },
    getCondaEnvs: () => ipcRenderer.invoke('get-conda-envs')
});
