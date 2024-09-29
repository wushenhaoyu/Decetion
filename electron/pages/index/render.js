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
    action_info.style.display = 'block'
    action_info_text.innerText =  "散射增强模块初始化中"
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
const action_info = document.getElementById("action_info");
const action_info_text = document.getElementById("action_info_text");
let progress =  0 
window.electronAPI.onBatStatus((statusMessage) => {
    // 处理模块状态消息
    action_info_text.innerText = dealWithProgress(statusMessage)
});
progress_dict = [
    "散射增强模块初始化中",
    "弱光增强模块初始化中",
    "行人检测初始化中",
    "车辆检测初始化中",
    "行人属性初始化中",
    "车辆属性初始化中",
    "行人追踪初始化中",
    "车辆追踪初始化中", 
    "初始化完毕",
    "正在进入"
]
dealWithProgress = (data) => {
    console.log(data)
    if (data == 1){
        progress += 1
    }
    else if (data == 2){
        progress = 0
        action_info.style.display = 'none'
    }
    else if (data == 3){
        progress += 1
        setTimeout(() => {
            console.log('延时')
            document.getElementById("action_info").style.display = 'none'
            progress = 0
            window.electronAPI.enterPage()
        }, 3000);
    }
    return progress_dict[progress]
}