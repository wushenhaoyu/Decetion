<template>
  <div style="width: 100%; height: 100%">
    <!-- 固定在右边的抽屉 -->
    <div :class="drawer_class_ctrl">
      <div class="drawer-content">
        <el-menu :default-active="activeIndex" class="el-menu-vertical">
          <el-menu-item index="1">处理中心</el-menu-item>
        </el-menu>
      </div>
      <div class="drawer-button-bar" @click="toggleDrawer">
        <!-- 小拉手按钮，点击它会调用 toggleDrawer 方法 -->
        <i :class="drawer_button_class_ctrl"></i>
      </div>
    </div>

    <!-- 以下为主内容 -->
    <div style="height: 5%"></div>
    <div class="main">
      <div style="height: 100%; width: 70%">
        <div style="height: 80%; width: 100%" class="upload">
          <!-- 上传视频  -->
          <el-upload
            class="upload-demo"
            drag
            :action="uploadUrl"
            multiple
            :before-upload="handleBeforeUpload"
            :on-success="handleSuccess"
            :on-error="handleError"
            :on-progress="handleProgress"
            v-if="!isShowVideo"
          >
            <div
              v-if="!showProgress"
              style="
                position: relative;
                top: 50%;
                left: 50%;
                transform: translate3d(-50%, -50%, 0);
                width: auto;
                height: auto;
              "
            >
              <i class="el-icon-upload"></i>
              <div
                class="el-upload__text"
                style="height: auto; line-height: 10vh"
              >
                将视频拖到此处，或<em>点击上传</em>
              </div>
            </div>
            <div
              v-if="showProgress"
              style="
                position: relative;
                top: 50%;
                left: 50%;
                transform: translate3d(-50%, -50%, 0);
                width: auto;
                height: auto;
              "
            >
              <el-progress
                v-if="showProgress"
                :percentage="progressPercentage"
                status="active"
              ></el-progress>
            </div>
          </el-upload>
          <!-- 上传视频  -->
          <!-- 显示视频  -->
          <video
            v-if="isShowVideo"
            id="example"
            class="vjs-default-skin vjs-big-play-centered"
            controls
            preload="auto"
          >
            <source style="background: #000" :src="videoUrl" type="video/mp4" />
            <p class="vjs-no-js">
              To view this video please enable JavaScript, and consider
              upgrading to a web browser
            </p>
          </video>
          <!-- 显示视频  -->
        </div>
        <div style="height: 5%"></div>
        <!-- 视频下方操作按钮-->
        <div
          class="bottom-ctrl"
          style="height: 15%; width: 100%; font-size: 2vw"
        >
          <div class="bottom-ctrl-one">
            <!-- <el-button type="primary" class="bottom-button" @click="switchCamera">{{isShowCamera ? '关闭摄像头' : '开启摄像头'}}</el-button>
                <el-button type="primary" class="bottom-button">开始检测</el-button> -->
            <!-- <el-button type="primary" class="bottom-button">开启录制</el-button> -->
            <el-button type="primary" class="bottom-button" @click="getVideo"
              >开始检测</el-button
            >
            <el-button type="primary" class="bottom-button" @click="resetVideo"
              >重置视频</el-button
            >
          </div>
        </div>
        <!-- 视频下方操作按钮-->
      </div>

      <div style="height: 100%; width: 30%">
        <div class="right-log">
          <div class="right-log-head">检测日志</div>
          <div>
            <el-input
              style="width: 100%; white-space: pre-line; "
              type="textarea"
              :autosize="{ minRows: 30, maxRows: 30 }"
              placeholder="请输入内容"
              v-model="textarea"
            >
            </el-input>
          </div>
        </div>
      </div>
    </div>
  </div>
  <!-- 以上为主内容 -->
</template>

<script>
import myvideo from "../../video/myvideo.vue";
export default {
  components: {
    myvideo,
  },
  data() {
    return {
      isShowVideo: false,
      drawerVisible: false,
      activeIndex: "4", // 更新为菜单项的实际索引
      videoUrl: "",
      uploadUrl: "",
      showProgress: false,
      progressPercentage: 0,
    };
  },
  computed: {
    drawer_class_ctrl() {
      return this.drawerVisible ? "drawer-open" : "drawer-close";
    },
    drawer_button_class_ctrl() {
      return [
        this.drawerVisible ? "drawer-button-open" : "drawer-button-close",
        this.drawerVisible ? "el-icon-caret-right" : "el-icon-caret-left",
      ];
    },
  },
  methods: {
    resetVideo() {
      this.isShowVideo = false;
    },
    getVideo() {
      this.isShowVideo = True;
      this.$axios
        .post("http://localhost:8000/getvideo/")
        .then((response) => {
          console.log(response);
        })
        .catch((error) => {
          console.error(error);
        });
    },
    toggleDrawer() {
      this.drawerVisible = !this.drawerVisible;
    },
    handleBeforeUpload(file) {
      // 检查视频格式
      const supportedFormats = ["mp4", "avi", "mov", "mkv"];
      const fileExtension = file.name.split(".").pop().toLowerCase();
      const isSupportedFormat = supportedFormats.includes(fileExtension);

      // 检查文件大小
      const maxSize = 100 * 1024 * 1024; // 100 MB
      const isSizeValid = file.size <= maxSize;

      if (isSupportedFormat && isSizeValid) {
        console.log("Valid video format and size");
        return true;
      } else {
        let errorMessage = "";
        if (!isSupportedFormat) {
          errorMessage += "不支持的视频格式。";
        }
        if (!isSizeValid) {
          errorMessage += "文件大小超过 100 MB。";
        }
        this.$message.error(errorMessage);
        return false;
      }
    },
    handleSuccess(response, file, fileList) {
      console.log("上传成功:", response);
      this.showProgress = false;
      this.progressPercentage = 0;
      this.$message.success("上传成功");
    },
    handleError(error, file, fileList) {
      console.error("上传失败:", error);
      this.showProgress = false;
      this.progressPercentage = 0;
      this.$message.error("上传失败");
    },
    handleProgress(event, file, fileList) {
      this.showProgress = true;
      this.progressPercentage = event.percent;
    },
  },
};
</script>

<style scoped>
.right-log {
  width: 80%;
  height: 100%;
  background-color: white;
  margin-left: 10%;
  margin-right: 10%;
  border-radius: 8px;
}
.right-log-head {
  height: 5vh;
  width: 100%;
  background-color: rgb(66, 159, 255);
  color: white;
  font-size: 1.5vw;
  line-height: 5vh;
  border-top-right-radius: 8px; /* 可选: 让边角变圆 */
  border-top-left-radius: 8px;
}
.bottom-ctrl {
  border-radius: 8px; /* 可选: 让边角变圆 */
  background-color: white;
  margin-top: 2vh;
  margin-bottom: 4vh;
  display: flex; /* 启用 Flexbox */
  flex-direction: column; /* 设置主轴方向为纵向 */
  justify-content: space-evenly; /* 在主轴上均匀分配子元素的间距 */
}
.bottom-ctrl-one {
  width: 100%;
  display: flex;
  justify-content: space-between;
}
.bottom-button {
  height: 8vh;
  width: 15vw;
  font-size: 1.5vw;
  margin-left: 0.5vw;
  margin-right: 0.5vw;
}
.upload {
  display: flex;
  flex-direction: column; /* 垂直方向排列子元素 */
  justify-content: center; /* 垂直方向居中对齐 */
  align-items: center; /* 水平方向居中对齐 */
  height: 100%; /* 确保容器有足够的高度 */
  width: 100%; /* 确保容器有足够的宽度 */
  border: 1px dashed #dcdfe6; /* 可选: 只是为了视觉效果 */
  border-radius: 8px; /* 可选: 让边角变圆 */
  margin-bottom: -2vh;
}
.upload div {
  width: 100%;
  height: 100%;
}
.upload .el-icon-upload {
  font-size: 20vw; /* 图标的大小 */
  margin-bottom: 3vh; /* 图标与文本之间的间距 */
}

.upload .el-upload__text {
  text-align: center; /* 确保文本内容在其容器中居中对齐 */
  font-size: 3vw;
}
/deep/ .el-upload {
  width: 100%;
  height: 100%;
}
/deep/ .el-upload .el-upload-dragger {
  width: 100%;
  height: 100%;
}

.main {
  display: flex;
  height: 90%;
  width: 100%;
  justify-content: space-evenly;
}
.drawer-open {
  position: fixed;
  top: 7vh;
  right: 0;
  width: 15vw;
  height: 100%;
  background-color: #fff;
  box-shadow: -1px 0 2px rgba(0, 0, 0, 0.5);
  z-index: 1000;
  transform: translateX(0);
  transition: transform 0.3s ease;
}

.drawer-close {
  position: fixed;
  top: 7vh;
  right: 0;
  width: 15vw;
  height: 100%;
  background-color: #fff;
  box-shadow: -1px 0 2px rgba(0, 0, 0, 0.5);
  z-index: 1000;
  transform: translateX(14vw);
  transition: transform 0.3s ease;
}

.drawer-content {
  height: 100%;
  overflow-y: auto;
}

.drawer-button-bar {
  background-color: white;
  height: 100%;
  width: 1vw;
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
}

.drawer-button-open {
  position: absolute;
  top: 50%;
  left: 100%; /* 将按钮放在抽屉外侧 */
  transform: translate3d(-100%, -50%, 0);
  transition: all 0.5s;
}

.drawer-button-close {
  position: absolute;
  top: 50%;
  left: 100%; /* 将按钮放在抽屉外侧 */
  transform: translate3d(-100%, -50%, 0) rotateX(180deg);
  transition: all 0.5s;
}
video::-webkit-media-controls-play-button {
  display: none;
}
video::-webkit-media-controls-timeline {
  display: none;
}
</style>
