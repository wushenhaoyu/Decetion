<template>
  <div style="width: 100%; height: 100%">
    <!-- 固定在右边的抽屉 -->
    <div :class="drawer_class_ctrl" style="width: 15vw">
      <div class="drawer-content" >
        <div class="right-log-head"style="line-height: 6vh;position: absolute;z-index: 5  ;height: 6vh;">处理中心</div>
        <div style="height: 3vh;"></div>
        <el-divider></el-divider>
        <div style="user-select:none;">
          <div style="height: 4vh;line-height: 4vh;  user-select:none;">散射增强</div>
          <el-switch
            v-model="haze"
            active-text="开启"
            inactive-text="关闭"
            @change="sendParameters"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select:none;">
          <div style="height: 4vh;line-height: 4vh;  user-select:none;">弱光增强</div>
          <el-switch
            v-model="dark"
            active-text="开启"
            inactive-text="关闭"
            @change="sendParameters"
          > </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select:none;">
          <div style="height: 4vh;line-height: 4vh;  user-select:none;">行人跟踪</div>
          <el-switch
            v-model="people_tracker_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="sendParameters"
          > </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select:none;">
          <div style="height: 4vh;line-height: 4vh;  user-select:none;">行人属性检测</div>
          <el-switch
            v-model="people_attribute_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="sendParameters"
          > </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select:none;">
          <div style="height: 4vh;line-height: 4vh;  user-select:none;">车辆跟踪</div>
          <el-switch
            v-model="vehicle_tracker_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="sendParameters"
          > </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select:none;">
          <div style="height: 4vh;line-height: 4vh;  user-select:none;">车辆属性检测 </div>
          <el-switch
            v-model="vehicle_attribute_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="sendParameters"
          > </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select:none;">
        <div style="height: 4vh;line-height: 4vh;  user-select:none;">车牌检测 </div>
          <el-switch
            v-model="vehicle_license_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="sendParameters"
          > </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select:none;">
        <div style="height: 4vh;line-height: 4vh;  user-select:none;">违章检测 </div>
          <el-switch
            v-model="vehicle_press_detector_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="sendParameters"
          > </el-switch>
      </div>
      <el-divider></el-divider>
        <div>
        <div style="height: 4vh;line-height: 4vh;  user-select:none;">违停检测 </div>
          <el-switch
            v-model="vehicle_invasion_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="sendParameters"
          > </el-switch>
        </div>

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
          <div v-if="isShowVideo" style="background-color: #000;width: 100%;height: 100%;">
            <video
            v-if="isShowVideo"
            controls
            preload="auto"
            style="background: #000;max-width: 100%; max-height: 100%; object-fit: contain;"
            :src="VideoUrl_"
            type = "video/mp4"
          >
          </video>
          <video
            v-if="!isShowVideo"
            id="example"
            controls
            preload="auto"
            style="background: #000;max-width: 100%; max-height: 100%; object-fit: contain;"
            :src="VideoUrl" type="video/mp4" 
          >
          </video>
          </div>
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
            <el-button type="primary" class="bottom-button" @click="dealWithVideo" 
              >开始检测</el-button
            >
            <el-button type="primary" class="bottom-button" @click="saveVideo"
              >导出视频</el-button
            >
            <el-button type="primary" class="bottom-button" 
              >截取图片</el-button
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
      haze: false,
      dark: false,
      people_detector_enable: false, // 行人监测
      people_tracker_enable: false,
      people_attribute_enable: false,
      vehicle_detector_enable: false,//车辆监测
      vehicle_tracker_enable:false,
      vehicle_press_detector_enable: false,
      vehicle_license_enable: false,
      vehicle_attribute_enable: false,
      vehicle_invasion_enable:false,
      isShowVideo: false,
      drawerVisible: false,
      activeIndex: "4", // 更新为菜单项的实际索引
      videoUrl: "",
      uploadUrl: "http://localhost:8000/uploadVideo",
      showProgress: false,
      progressPercentage: 0,
      isShowLocalVideo: true,
      VideoUrl_:""
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
    VideoUrl() {
      return `http://localhost:8000/stream_photo?name=${this.VideoName}&style=2`;
    },
  },
  methods: {
    saveVideo() {
        if (!this.VideoName)
      {
        this.$message({
            type: 'error',
            message: '未上传视频'
          });
      }else{
        fetch(this.VideoUrl)
        .then(response => response.blob()) // 将响应转换为 Blob
        .then(blob => {
          // 创建一个临时 URL
          const url = URL.createObjectURL(blob);
          
          // 创建一个隐藏的 <a> 元素用于下载
          const a = document.createElement('a');
          a.style.display = 'none';
          a.href = url;
          a.download = 'downloaded-video.mp4'; // 下载时的文件名
          
          // 将 <a> 元素添加到 DOM 中并触发点击事件
          document.body.appendChild(a);
          a.click();

          // 释放 URL 对象并移除 <a> 元素
          URL.revokeObjectURL(url);
          document.body.removeChild(a);
        })
        .catch(error => {
          this.$message({
            type: 'error',
            message: '未进行预测'
          });
        });
      }
    },
    sendParameters(value) {
      this.checkParameter(value)
      let data = {
        haze: this.haze,
        dark: this.dark,
        hdr:  false,
        people_detector: this.people_detector_enable,
        people_tracker: this.people_tracker_enable,
        people_attr_detector: this.people_attribute_enable,
        vehicle_tracker: this.vehicle_tracker_enable,
        vehicle_detector: this.vehicle_detector_enable,
        vehicle_attr_detector: this.vehicle_attribute_enable,
        vehicleplate_detector: this.vehicle_license_enable,
        vehicle_press_detector: this.vehicle_press_detector_enable,
        vehicle_invasion:this.vehicle_invasion_enable
      }
      this.$axios.post('http://localhost:8000/ConfirmParams', data).then(res => {
      })
    },
    checkParameter(value) {
      var that = this;
      if (value) { // 有东西开启了，要保证额外功能开启的时候，保证追踪或者检测开启
        const people_list = [that.people_attribute_enable];
        for (let i = 0; i < people_list.length; i++) {
          if (people_list[i]) {
            that.people_tracker_enable = true;
          }
        }
        const vehicle_list = [that.vehicle_attribute_enable, that.vehicle_license_enable, that.vehicle_press_detector_enable, that.vehicle_invasion_enable];
        for (let i = 0; i < vehicle_list.length; i++) {
          if (vehicle_list[i]) {
            that.vehicle_tracker_enable = true;
          }
        }
      } else { // 检测关闭，额外功能也要关闭
        if (!(that.people_detector_enable && that.people_tracker_enable)) {
          that.people_attribute_enable = false;
        }
        if (!(that.vehicle_detector_enable && that.vehicle_tracker_enable)) {
          that.vehicle_attribute_enable = false;
          that.vehicle_license_enable = false;
          that.vehicle_press_detector_enable = false;
          that.vehicle_invasion_enable = false;
        }
      }
    },
    resetVideo() {
      this.isShowVideo = false;
      this.isShowLocalVideo = true;
      this.videoName = "";
      this.videoUrl_ = "";
    },
    dealWithVideo()
    {
        data = {
          name: this.videoName
        }
         this.$axios.post("http://localhost:8000/start_process_video",data).then(res => {
        this.getVideo();
      });
    },
    getVideo() {
      this.isShowLocalVideo = false;
      this.$nextTick(() => {
        this.isShowVideo = true;
      });
    },
    toggleDrawer() {
      this.drawerVisible = !this.drawerVisible;
    },
    handleBeforeUpload(file) {
      const supportedFormats = ["mp4", "avi", "mov", "mkv"];
      const fileExtension = file.name.split(".").pop().toLowerCase();
      const isSupportedFormat = supportedFormats.includes(fileExtension);

      const maxSize = 100 * 1024 * 1024; // 100 MB
      const isSizeValid = file.size <= maxSize;

      if (isSupportedFormat && isSizeValid) {
        this.showConfirmDialog(file);
        return false; // 阻止自动上传
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
    showConfirmDialog(file) {
      this.$confirm('确认上传此文件？', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        // 用户点击确定按钮，允许上传
        this.uploadFile(file);
      }).catch(() => {
        // 用户点击取消按钮，取消上传
        this.$message({
          type: 'info',
          message: '已取消上传'
        });
      });
    },
    handleError(error, file) {
      console.error('上传失败:', error);
      this.$message.error("上传失败");
    },
    handleSuccess(response, file) {
      this.$message({
        type: 'success',
        message: '上传成功!'
      });
      const reader = new FileReader();
      reader.onload = (e) => {
        this.VideoUrl_ = e.target.result;
        this.isShowVideo = true
      };
      reader.readAsDataURL(file);
      // 处理成功逻辑
    },
    handleProgress(event, file, fileList) {
      // this.progressPercentage = event.percent;
      // this.showProgress = true;
      // console.log('上传进度:', event.percent);
    },
    uploadFile(file) {

      const formData = new FormData();
      formData.append('video', file);
    

      const config = {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        /*onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          this.progressPercentage = percentCompleted;
          this.showProgress = true;
        }*/
      }; 
      // this.startProgressPolling();
      console.log("123")
      this.$axios.post(this.uploadUrl, formData, config).then(response => {
        this.handleSuccess(response, file);
        this.videoName = response.data.videoname;
      }).catch(error => {
        this.handleError(error, file);
      });
      /*this.$axios.post(this.uploadUrl, formData, config).then(response => {
        this.videoName = response.data.videoname; // 假设服务器返回的数据结构为 { video_name: 'some-video-name' }
        console.log('Video name:', this.videoName);
        //this.startProgressPolling(); // 开始轮询进度
        this.handleSuccess(response, file);
      }).catch(error => {
        this.handleError(error, file);
      });*/
    },
    startProgressPolling() {
      data = {
          name: this.videoName
      }
      this.$axios.post("http://127.0.0.1:8000/get_progress",data).then(res => {
        this.progress = res.progress;
      });
      const loading = this.$loading({
          lock: true,
          text: thisprogress,
          spinner: 'el-icon-loading',
          background: 'rgba(0, 0, 0, 0.7)'
        });
        setTimeout(() => {
          loading.close();
        }, 2000);
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
  top: 10vh;
  right: 0;
  width: 15vw;
  height: 100%;
  background-color: #fff;
  box-shadow: -1px 0 2px rgba(0, 0, 0, 0.5);
  z-index: 1000;
  transform: translateX(0);
  transition: transform 0.3s ease;
  border-top-right-radius: 8px; /* 可选: 让边角变圆 */
  border-top-left-radius: 8px;
  user-select:none;
}

.drawer-close {
  position: fixed;
  top: 10vh;
  right: 0;
  width: 15vw;
  height: 100%;
  background-color: #fff;
  box-shadow: -1px 0 2px rgba(0, 0, 0, 0.5);
  z-index: 1000;
  transform: translateX(14vw);
  transition: transform 0.3s ease;
  border-top-right-radius: 8px; /* 可选: 让边角变圆 */
  border-top-left-radius: 8px;
  user-select:none;
}

.drawer-content {
  height: 100%;
  overflow-y: auto;
}
.drawer-content::-webkit-scrollbar {
  display: none; /* 针对 Chrome, Safari, Edge 浏览器隐藏滚动条 */
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
/*video::-webkit-media-controls-play-button {
  display: none;
}
video::-webkit-media-controls-timeline {
  display: none;
}*/
</style>
