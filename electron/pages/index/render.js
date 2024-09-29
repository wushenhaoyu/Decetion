console.log('render');

document.addEventListener('DOMContentLoaded', function() {
    // 获取所有选项元素
    const options = document.querySelectorAll('.left-font');
    const childpage = document.querySelectorAll('.child');

    // 添加点击事件监听器
    options.forEach((option, index) => {
        option.addEventListener('click', function() {
            // 移除所有选项的高亮状态
            options.forEach(opt => opt.classList.remove('highlight'));
            childpage.forEach(child => child.style.display = 'none');
            // 给当前点击的选项添加高亮状态
            this.classList.add('highlight');
            childpage[index].style.display = 'block';
        });
    });

    // 默认高亮首页并加载首页内容
    options[0].classList.add('highlight');
    childpage[0].style.display = 'block';
});

// 处理运行 BAT 文件按钮点击事件
const button = document.getElementById("action");
button.onclick = async function() {
    const result = await window.electronAPI.runBatFile();
    if (result.success) {
        console.log("BAT 文件运行成功");
    }
};

// 输出区域
const output = document.getElementById("output");

// 监听来自主进程的输出信息
window.electronAPI.onBatOutput((data) => {
    output.innerHTML += data;  // 将输出追加到输出区域
    output.scrollTop = output.scrollHeight;  // 自动滚动到最新输出
});
