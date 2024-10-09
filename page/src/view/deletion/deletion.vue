<template>
    <div style="width: 100%; height: 100%;background-color: white;" >
        <div style="font-size: 2vw;line-height: 12vh;font-weight: 900;">🍈🍉缓存清理🍓🍇</div>
        <div >
            <el-alert
            style="width: 96%;margin-left: 2vw;height: 10vh;"
            type="warning"
            :closable="false"
            >
            <div style="font-size: 1vw;font-weight: 900;text-align: left;">温馨提示</div>
            <div style="font-size: 1vw;font-weight: 500;">请留意各模块存储缓存大小，及时清理！</div>
            </el-alert>
            <div  style="width: 96%;margin-left: 2vw;">
                <div style="height: 1vh;"></div>
                <el-divider content-position="left" style="font-size: 1.5vw;">
                    <div class="my_font">摄像头录像缓存</div>
                </el-divider>
                <div style="height: 1vh;"></div>
                <div style="display: flex;height: 3vh;">
                    <div style="font-size: 0.8vw;line-height: 4vh;font-weight: 900;">缓存大小:</div>
                    <div style="width: 2%;"></div>
                    <el-input style="width: 8%;height: 3vh;"
                    placeholder="大小"
                    v-model="input"
                    :disabled="true">
                    </el-input>
                    <div style="width: 2%;"></div>
                    <el-button type="danger" class="my_button">清理</el-button>
                </div>
                <div style="height: 1vh;"></div>
                <div style="height: 1vh;"></div>
                <el-divider content-position="left" style="font-size: 1.5vw;">
                    <div class="my_font">视频处理缓存</div>
                </el-divider>
                <div style="height: 1vh;"></div>
                <div style="height: 1vh;"></div>
                <div style="display: flex;height: 3vh;">
                    <div style="font-size: 0.8vw;line-height: 4vh;font-weight: 900;">缓存大小:</div>
                    <div style="width: 2%;"></div>
                    <el-input style="width: 8%;height: 3vh;"
                    placeholder="大小"
                    v-model="input"
                    :disabled="true">
                    </el-input>
                    <div style="width: 2%;"></div>
                    <el-button type="danger" class="my_button">清理</el-button>
                </div>
                <div style="height: 1vh;"></div>
                <div style="height: 1vh;"></div>
                <el-divider content-position="left" style="font-size: 1.5vw;">
                    <div class="my_font">图像处理缓存</div>
                </el-divider>
                <div style="height: 1vh;"></div>
                <div style="display: flex;height: 3vh;">
                    <div style="font-size: 0.8vw;line-height: 4vh;font-weight: 900;">缓存大小:</div>
                    <div style="width: 2%;"></div>
                    <el-input style="width: 8%;height: 3vh;"
                    placeholder="大小"
                    v-model="input"
                    :disabled="true">
                    </el-input>
                    <div style="width: 2%;"></div>
                    <el-button type="danger" class="my_button">清理</el-button>
                </div>
                <div style="height: 1vh;"></div>
            </div>
        </div>

      </div>
      <!-- 以上为主内容 -->
</template>
<script>
  export default {
    data() {
      return {
        
      }
    },
  }
</script>
<style>
.my_font{
    font-size: 1.1vw;
    font-weight: 600;
}
.my_button {
  height: 4vh;
  width: 6vw;
  font-size: 0.8vw;
  margin-left: 0.5vw;
  margin-right: 0.5vw;
  font-weight: 900;
}
</style>