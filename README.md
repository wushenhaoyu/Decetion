# 仓库结构
```bash
│                                                
├───AIDjango          # 系统后端（Django）
│                                        
├───dark              # 弱光增强
│
├───electron          # 启动器（Electron）
│
├───haze              # 散射增强
│
├───hdr               # 高动态范围修复
│
├───my_detection      # paddledetection魔改
│
└───page              # Vue.js 前端
```   
# 运行

## Django
疯狂运行，报啥错安啥。有些库比较复杂，使用 Python 3.8，具体版本可查看 `requirements.txt`。注意有些库可能需要 C 编译安装。

```bash
python AIdjango/manage.py runserver
```

## Vue

1.进入 page 目录：
``` bash
cd page
```

2. 安装 npm 依赖：
```bash
npm install
```

3. 启动开发环境：
```bash
npm run dev
```
