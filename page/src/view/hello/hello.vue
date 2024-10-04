<template>
    <div style="height: 100%;width: 100%;">
        <div class="bg">
            <div style="display: flex;">
                <div style="width: 50%;height: 100%;">
                    <div style="display: flex;justify-content: center">
                    <div style="display: flex;flex-direction: column;align-items: flex-start;">
                        <div style="display: flex;align-items: flex-end;">
                            <div style="height: 15vh;"></div>
                            <div style="color: #1527c2;font-size: 5vw;font-weight: 600;font-style: italic;line-height: 7vh;">Paddle</div>
                            <div style="color: #3c3c43;font-size: 3.5vw;font-weight: 600;line-height: 5vh;">&nbsp;路况瞭望</div>
                        </div>
                        <div style="color: #3c3c43;font-size: 3vw;font-weight: 900;">复杂天气路况精准检测</div>
                        <div style="color: #3c3c43;font-size: 1.5vw;font-weight: 500;line-height: 8vh;">基于Paddle + PPdet + Vue + Django</div>
                        <div style="display: flex;justify-content: flex-start   ;">
                        <el-button type="primary" round style="font-weight: 700;" @click="goAIstudio">AIstudio</el-button>
                        <el-button type="info" round style="font-weight: 700;" @click="goToGithub">Github</el-button>
                        <el-button type="success" round style="font-weight: 900;" @click="rate" >评价一下</el-button>
                    </div>
                    </div>
                    </div>
                    
                        <div style="width: 100%;height: 40%;">
                            <div style="width: 100%;height: 2vh"> </div>
                            <div style="display: flex;justify-content: space-evenly;">
                                <div style="width: 30%;height: 100%;background-color: #f6f6f7;border-radius: 1.5vw;">
                                    <div style="display: flex;"><div style="background-color: #e3e3e5;border-radius: 0.5vw;margin-left: 1vw;margin-top: 1vw;"><img style="width: 6vh;" src="../../assets/1.png"/></div></div>
                                    <div style="text-align: left;padding-left: 1.5vw;font-weight: 900;line-height: 4vh;font-size: 1vw;">复杂天气条件</div>
                                    <div style="text-align: left;padding-left: 1.5vw;padding-right: 1.5vw;font-weight: 700;color: #6b6b70;font-size: 0.8vw;line-height: 2vh;">集成弱光、散射、HDR修复三种增强网络</div>
                                    <div style="height: 1vh;"></div>
                                </div>
                                <div style="width: 30%;height: 100%;background-color: #f6f6f7;border-radius: 1.5vw;">
                                    <div style="display: flex;"><div style="background-color: #e3e3e5;border-radius: 0.5vw;margin-left: 1vw;margin-top: 1vw;"><img style="width: 6vh;" src="../../assets/1.png"/></div></div>
                                    <div style="text-align: left;padding-left: 1.5vw;font-weight: 900;line-height: 4vh;font-size: 1vw;">灵活功能调度</div>
                                    <div style="text-align: left;padding-left: 1.5vw;padding-right: 1.5vw;font-weight: 700;color: #6b6b70;font-size: 0.8vw;line-height: 2vh;">一次初始化，随意快速切换模型功能</div>
                                    <div style="height: 1vh;"></div>
                                </div>
                                <div style="width: 30%;height: 100%;background-color: #f6f6f7;border-radius: 1.5vw;">
                                    <div style="display: flex;"><div style="background-color: #e3e3e5;border-radius: 0.5vw;margin-left: 1vw;margin-top: 1vw;"><img style="width: 6vh;" src="../../assets/1.png"/></div></div>
                                    <div style="text-align: left;padding-left: 1.5vw;font-weight: 900;line-height: 4vh;font-size: 1vw;">PPDet优化</div>
                                    <div style="text-align: left;padding-left: 1.5vw;padding-right: 1.5vw;font-weight: 700;color: #6b6b70;font-size: 0.8vw;line-height: 2vh;">二次封装PaddleDetection,便捷化模型推理</div>
                                    <div style="height: 1vh;"></div>
                                </div>
                            </div>
                            <div style="width: 100%;height: 2vh"> </div>
                            <div style="display: flex;justify-content: space-evenly;">
                                <div style="width: 30%;height: 100%;background-color: #f6f6f7;border-radius: 1.5vw;">
                                    <div style="display: flex;"><div style="background-color: #e3e3e5;border-radius: 0.5vw;margin-left: 1vw;margin-top: 1vw;"><img style="width: 6vh;" src="../../assets/1.png"/></div></div>
                                    <div style="text-align: left;padding-left: 1.5vw;font-weight: 900;line-height: 4vh;font-size: 1vw;">轻量化网络结构</div>
                                    <div style="text-align: left;padding-left: 1.5vw;padding-right: 1.5vw;font-weight: 700;color: #6b6b70;font-size: 0.8vw;line-height: 2vh;">弱光、散射增强网络采用超轻量化架构</div>
                                    <div style="height: 1vh;"></div>
                                </div>
                                <div style="width: 30%;height: 100%;background-color: #f6f6f7;border-radius: 1.5vw;">
                                    <div style="display: flex;"><div style="background-color: #e3e3e5;border-radius: 0.5vw;margin-left: 1vw;margin-top: 1vw;"><img style="width: 6vh;" src="../../assets/1.png"/></div></div>
                                    <div style="text-align: left;padding-left: 1.5vw;font-weight: 900;line-height: 4vh;font-size: 1vw;">可边缘计算部署</div>
                                    <div style="text-align: left;padding-left: 1.5vw;padding-right: 1.5vw;font-weight: 700;color: #6b6b70;font-size: 0.8vw;line-height: 2vh;">网络复杂度低，算力资源要求小，支持串口视频流输入</div>
                                    <div style="height: 1vh;"></div>
                                </div>
                                <div style="width: 30%;height: 100%;background-color: #f6f6f7;border-radius: 1.5vw;">
                                    <div style="display: flex;"><div style="background-color: #e3e3e5;border-radius: 0.5vw;margin-left: 1vw;margin-top: 1vw;"><img style="width: 6vh;" src="../../assets/emoji_u23f0.svg"/></div></div>
                                    <div style="text-align: left;padding-left: 1.5vw;font-weight: 900;line-height: 4vh;font-size: 1vw;">实时性高</div>
                                    <div style="text-align: left;padding-left: 1.5vw;padding-right: 1.5vw;font-weight: 700;color: #6b6b70;font-size: 0.8vw;line-height: 2vh;">模型推理迅速，可保持较高帧率</div>
                                    <div style="height: 1vh;"></div>
                                </div>
                            </div>
                        </div>
                </div>
                <div style="width: 50%;height: 100%;display: flex;justify-content: flex-end;">
                    <img src="../../assets/bg.jpg" style="height: 80%;width: 100%; border-top-right-radius: 1.5vw; border-bottom-right-radius: 2.5vw;"/>
                </div>
            </div>
            <div v-if="showAlert" style="box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06);background-color: white;width: 15%;height:20%;position: absolute;left: 50%;top:50%;transform: translate(-50%,-50%);z-index: 5;border-radius: 0.5vw;">
                <div style="height: 20%;"></div>
                <div v-if="value2" style="height: 20%;">感谢您的评价！</div>
                <el-rate
                    v-model="value2"
                    :colors="colors">
                </el-rate>
                <div style="height: 10%;"></div>
                <el-button type="primary" round @click="rate_">确定</el-button>
            </div>
        </div>
    </div>
</template>

<style>     
    .bg{
        position: absolute; width: 90%;height: 80%;background-color: white;left: 52%;top:50%;transform: translate(-50%,-50%);border-radius: 1.5vw;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06);
    }
</style>
<script>
    export default {
        data() {
            return {
                showAlert:  false, // 控制 Alert 显示与否
                value2: null,
                colors: ['#99A9BF', '#F7BA2A', '#FF9900'] 
            }
        },
        methods: {
            goToGithub(){
                window.open("https://github.com/wushenhaoyu/Decetion.git")
            },
            goAIstudio(){
                window.open("https://aistudio.baidu.com/projectdetail/8319014")
            },
            rate() {
                this.showAlert = true
            },
            rate_(){
                this.showAlert = false
            }
        }
    }
</script></script>