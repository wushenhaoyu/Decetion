console.log("hello world");
const button = document.getElementById("action");
button.onclick = async function() {
    try {
        const result = await window.electronAPI.action();  // 确保是异步调用
        console.log(result);  // 打印从主进程返回的结果
    } catch (error) {
        console.error('Error in action:', error);  // 捕获错误
    }
};