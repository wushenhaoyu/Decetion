const { contextBridge, ipcRenderer } = require('electron');

// 暴露给渲染进程的 API
contextBridge.exposeInMainWorld('electron', {
    // 获取 Conda 环境列表
    getCondaEnvironments: () => ipcRenderer.invoke('get-conda-environments'),
    
    // 发送选择的 Conda 环境给主进程
    selectEnvironment: (condaEnv) => ipcRenderer.send('selected-environment', condaEnv)
});
