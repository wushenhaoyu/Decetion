<template>
  <div style="width: 100%; height: 100%">
    <!-- 固定在右边的抽屉 -->
    <div :class="drawer_class_ctrl" style="width: 15vw">
      <div class="drawer-content">
        <div
          class="right-log-head"
          style="line-height: 6vh; position: absolute; z-index: 5; height: 6vh"
        >
          处理中心
        </div>
        <div style="height: 3vh"></div>
        <el-divider></el-divider>
        <div style="user-select: none">
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            散射增强
          </div>
          <el-switch
            v-model="haze"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select: none">
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            弱光增强
          </div>
          <el-switch
            v-model="dark"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select: none">
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            高动态范围修复
          </div>
          <el-switch
            v-model="hdr"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select: none">
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            行人检测
          </div>
          <el-switch
            v-model="people_detector_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select: none">
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            行人属性检测
          </div>
          <el-switch
            v-model="people_attribute_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select: none">
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            车辆检测
          </div>
          <el-switch
            v-model="vehicle_detector_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select: none">
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            车辆属性检测
          </div>
          <el-switch
            v-model="vehicle_attribute_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select: none">
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            车牌检测
          </div>
          <el-switch
            v-model="vehicle_license_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div style="user-select: none">
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            违章检测
          </div>
          <el-switch
            v-model="vehicle_press_detector_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
        </div>
        <el-divider></el-divider>
        <div>
          <div style="height: 4vh; line-height: 4vh; user-select: none">
            违停检测
          </div>
          <el-switch
            v-model="vehicle_invasion_enable"
            active-text="开启"
            inactive-text="关闭"
            @change="checkParameter"
          >
          </el-switch>
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
          <!-- 上传图片  -->
          <el-upload
            class="upload-demo"
            drag
            :action="uploadUrl"
            multiple
            :before-upload="handleBeforeUpload"
            :on-success="handleSuccess"
            :on-error="handleError"
            :on-progress="handleProgress"
            v-if="!isShowPhoto"
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
                将图片拖到此处，或<em>点击上传</em>
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
          <!-- 上传图片  -->
          <!-- 显示图片  -->
          <div
            v-if="isShowPhoto"
            style="
              background-color: #000;
              width: 100%;
              height: 100%;
              display: flex;
              justify-content: center;
              align-items: center;
            "
            v-images-loaded:on.progress="imageProgress"
          >
            <img
              v-if="isShowLocalPhoto"
              id="example"
              class="vjs-default-skin vjs-big-play-centered"
              :src="photoUrl_"
              alt="Image"
              style="
                background: #000;
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
              "
            />
            <img
              v-if="!isShowLocalPhoto && photoUrl"
              id="example"
              class="vjs-default-skin vjs-big-play-centered"
              :src="photoUrl"
              :key="photoName"
              alt="Image"
              style="
                background: #000;
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
              "
            />
            <!--提示部分-->
          <div v-if="logDetail"
     ref="draggableBox"
     style="height: 80%; width: 60%; position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%); z-index: 8; background-color: white; border-radius: 0.5vw;">
  <!-- 顶部标题 -->
  <div @mousedown="startDrag"
       style="cursor: move; width: 100%; height: 7vh; border-top-left-radius: 0.5vw; border-top-right-radius: 0.5vw; background-color: rgb(66,159,255); text-align: center; line-height: 7vh; font-size: 2vw; color: white; font-weight: 900;">
    检测图片

    <!-- 关闭按钮 -->
    <button @click="closeWindow" 
            style="position: absolute; right: 10px; top: 10px; background: transparent; border: none; font-size: 2vw; color: white; cursor: pointer;">
      ×
    </button>
  </div>

  <!-- 主体内容，使用 flex 布局 -->
  <div style="display: flex; height: calc(80% - 7vh); padding: 10px;">

    <!-- 左侧图片区域 -->
    <div style="flex: 3; display: flex; justify-content: center; align-items: center;padding-top: 40px;" >
      <img :src="detailPhotoUrl" alt="image" style="width: 75%; height: auto; border-radius: 0.5vw;">
    </div>

    <!-- 右侧文本内容 -->
    <div style="flex: 1.3; padding: 20px; display: flex; flex-direction: column; justify-content: center; margin-left: -30px;">
  <div style="font-size: 1.8vw; font-weight: 700; margin-bottom: 10px;">检测信息</div>
  
  <!-- 检测类别 -->
  <div style="font-size: 1.8vw; color: gray; margin-bottom: 10px;">
    检测类别: <br>
    <span style="font-weight: normal;">{{ id }}</span>
  </div>

  <!-- 检测时间 -->
  <div style="font-size: 1.8vw; color: gray; margin-bottom: 10px;">
    检测时间: <br>
    <span style="font-weight: normal;">{{ time }}</span>
  </div>

  <!-- 目标坐标 -->
  <div style="font-size: 1.8vw; color: gray;">
    目标坐标: <br>
    <span style="font-weight: normal;">{{ location }}</span>
  </div>
</div>

  </div>
</div>

          <!--提示部分-->
          </div>

          <!-- 显示图片  -->
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
            <el-button
              type="primary"
              class="bottom-button"
              @click="dealwithPhoto"
              >开始检测</el-button
            >
            <el-button type="primary" class="bottom-button" @click="savePhoto"
              >导出图片</el-button
            >
            <el-button type="primary" class="bottom-button" @click="resetPhoto"
              >重置图片</el-button
            >
            <el-button type="primary" class="bottom-button" disabled
              >开发ing</el-button
            >
          </div>
        </div>
        <!-- 视频下方操作按钮-->
      </div>

      <div style="height: 100%; width: 30%">
        <div class="right-log">
          <div class="right-log-head">检测日志</div>
          <el-table
            ref="multipleTable"
            :data="paginatedData"
            tooltip-effect="dark"
            style="width: 100%"
            :row-style="{ height: '50px' }"
          >
            <el-table-column width="5"> </el-table-column>
            <el-table-column prop="id" label="类别" width="60">
              <!-- <template slot-scope="scope">{{ scope.row.date }}</template> -->
            </el-table-column>
            <el-table-column prop="time" label="时间" width="150">
            </el-table-column>
            <el-table-column prop="location" label="坐标" width="150">
            </el-table-column>
            <el-table-column label="操作" width="110">
              <template slot-scope="scope">
                <el-button
                  type="primary"
                  class="chaKanButton"
                  @click="handleButtonClick(scope.row.id,scope.row.time,scope.row.location, scope.row.name)"
                >
                  查看图片
                </el-button>
              </template>
            </el-table-column>
          </el-table>
          <div style="padding: 20px; text-align: center ;font-size: 3.5vw;">
            <el-pagination
              :current-page="pageNum"
              :page-sizes="[10, 20, 50]"
              :page-size="pageSize"
              layout="total, prev, pager, next, jumper"
              :total="total"
            >
            </el-pagination>
          </div>
        </div>
      </div>
    </div>
  </div>
  <!-- 以上为主内容 -->
</template>

<script>
import imagesLoaded from "vue-images-loaded";
export default {
  directives: {
    imagesLoaded,
  },
  data() {
    const item = {
      location: "50,50",
      id: "1234567",
      score: "100",
    };
    return {
      id : "",
      time : 0,
      location: "",
      isDragging: false, // 用于追踪拖动状态
      offsetX: 0,
      offsetY: 0,
      currentPage: 1,
      pageNum: 0,
      pageSize: 10,
      tableData: [],
      total: 0,
      haze: false,
      dark: false,
      hdr: false,
      people_detector_enable: false,
      people_tracker_enable: false,
      people_attribute_enable: false,
      vehicle_detector_enable: false,
      vehicle_tracker_enable: false,
      vehicle_press_detector_enable: false,
      vehicle_license_enable: false,
      vehicle_attribute_enable: false,
      vehicle_invasion_enable: false,
      isShowPhoto: false,
      drawerVisible: false,
      activeIndex: "4",
      showProgress: false,
      progressPercentage: 0,
      photoName: "",
      photoUrl_: "", //本地图片路径,
      isShowLocalPhoto: true,
      uploadUrl: "",
      photoUrl: "",
      detailPhotoName: "",
      logDetail: false,
    };
  },
  created() {
    // 在生命周期钩子里初始化 cameraUrl
    this.uploadUrl =  this.$apiBaseUrl + "/upload_photo";
  },
  computed: {
    
    paginatedData() {
      const start = (this.currentPage - 1) * this.pageSize;
      const end = this.currentPage * this.pageSize;
      return this.tableData.slice(start, end);
    },
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
  mounted() {
    this.total = this.tableData.length; // 设置总数据条目数
  },
  methods: {
    handleButtonClick(row) {
      // 处理按钮点击事件，例如打开图片等
      console.log('查看图片:', row.imageUrl);
      // 这里可以添加逻辑，比如打开一个对话框显示图片
    },
    handleSizeChange(){
      
    },
    handleCurrentChange(){

    },
    startDrag(event) {
      this.isDragging = true; // 开始拖动
      this.offsetX =
        event.clientX - this.$refs.draggableBox.getBoundingClientRect().left;
      this.offsetY =
        event.clientY - this.$refs.draggableBox.getBoundingClientRect().top;

      document.addEventListener("mousemove", this.doDrag);
      document.addEventListener("mouseup", this.stopDrag);
    },
    doDrag(event) {
      if (this.isDragging) {
        const left = event.clientX - this.offsetX;
        const top = event.clientY - this.offsetY;

        this.$refs.draggableBox.style.left = `${left}px`;
        this.$refs.draggableBox.style.top = `${top}px`;
        this.$refs.draggableBox.style.transform = "none"; // 移动时取消 `transform`
      }
    },
    stopDrag() {
      this.isDragging = false;
      document.removeEventListener("mousemove", this.doDrag);
      document.removeEventListener("mouseup", this.stopDrag);
    },
    closeWindow() {
      this.logDetail = false; // 设置为 false 以关闭窗口
    },
    async handleButtonClick(id, time, location, photoName) {
      console.log(id, photoName);
      this.id = id;
      this.time = time;
      this.location = location;
      this.detailPhotoName = photoName;
      if (id == "行人") {
        try {
          const response = await fetch(
             this.$apiBaseUrl + `/stream_photo?name=${this.detailPhotoName}&style=3`
          );

          if (!response.ok) {
            throw new Error("Network response was not ok");
          }

          console.log(response.url);

          this.detailPhotoUrl = response.url; // 将这个 URL 赋值给视频的 src

          console.log(this.detailPhotoUrl);
          this.logDetail = true;
        } catch (error) {
          console.error("There was a problem with the fetch operation:", error);
        }
      } else {
        try {
          const response = await fetch(
             this.$apiBaseUrl + `/stream_video?name=${this.detailPhotoName}&style=4`
          );

          if (!response.ok) {
            throw new Error("Network response was not ok");
          }

          console.log(response.url);

          this.videoUrl = response.url; // 将这个 URL 赋值给视频的 src

          console.log(this.videoUrl);
          this.isShowLocalVideo = false;
          this.isShowVideo = true; // 控制视频显示的变量
        } catch (error) {
          console.error("There was a problem with the fetch operation:", error);
        }
      }
    },
    savePhoto() {
      if (!this.photoName) {
        this.$message({
          type: "error",
          message: "未上传图片",
        });
      } else {
        fetch(this.photoUrl)
          .then((response) => response.blob()) // 将响应转换为 Blob
          .then((blob) => {
            // 创建一个临时 URL
            const url = URL.createObjectURL(blob);

            // 创建一个隐藏的 <a> 元素用于下载
            const a = document.createElement("a");
            a.style.display = "none";
            a.href = url;
            a.download = "downloaded-image.jpg"; // 下载时的文件名

            // 将 <a> 元素添加到 DOM 中并触发点击事件
            document.body.appendChild(a);
            a.click();

            // 释放 URL 对象并移除 <a> 元素
            URL.revokeObjectURL(url);
            document.body.removeChild(a);
          })
          .catch((error) => {
            this.$message({
              type: "error",
              message: "未进行预测",
            });
          });
      }
    },
    imageProgress(instance, image) {
      const result = image.isLoaded ? "loaded" : "broken";
    },
    sendParameters() {
      let data = {
        haze: this.haze,
        dark: this.dark,
        hdr: this.hdr,
        people_detector: this.people_detector_enable,
        people_tracker: this.people_tracker_enable,
        people_attr_detector: this.people_attribute_enable,
        vehicle_tracker: this.vehicle_tracker_enable,
        vehicle_detector: this.vehicle_detector_enable,
        vehicle_attr_detector: this.vehicle_attribute_enable,
        vehicleplate_detector: this.vehicle_license_enable,
        vehicle_press_detector: this.vehicle_press_detector_enable,
        vehicle_invasion: this.vehicle_invasion_enable,
      };
      return this.$axios
        .post( this.$apiBaseUrl+"/ConfirmParams", data)
        .then((res) => {
          
        });
    },
    checkParameter(value) {
      var that = this;
      if (value) {
        // 有东西开启了，要保证额外功能开启的时候，保证追踪或者检测开启
        const people_list = [that.people_attribute_enable];
        for (let i = 0; i < people_list.length; i++) {
          if (people_list[i]) {
            that.people_tracker_enable = true;
          }
        }
        const vehicle_list = [
          that.vehicle_attribute_enable,
          that.vehicle_license_enable,
          that.vehicle_press_detector_enable,
          that.vehicle_invasion_enable,
        ];
        for (let i = 0; i < vehicle_list.length; i++) {
          if (vehicle_list[i]) {
            that.vehicle_tracker_enable = true;
          }
        }
      } else {
        // 检测关闭，额外功能也要关闭
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
    async dealwithPhoto() {
  let loadingInstance; // 在 try 块外定义 loadingInstance，以便在 catch 中访问
      console.log("123");
      
  try {
    // 显示 loading
    loadingInstance = this.$loading({
      lock: true,
      text: `加载中...`, // 去掉百分比显示
      spinner: "el-icon-loading",
      background: "rgba(0, 0, 0, 0.7)",
    });

    // 发送处理参数
    await this.sendParameters();

    let data = {
      name: this.photoName,
    };
    console.log(data);

    // 开始处理照片
    await this.$axios.post( this.$apiBaseUrl+"/start_process_photo", data);


    // 获取照片和日志
    await this.getPhoto();
    // await this.getLog();
  } catch (error) {
    console.error("处理过程出错:", error);
  } finally {
    // 确保 loading 被关闭
    if (loadingInstance) {
      loadingInstance.close();
    }
  }
},


    resetPhoto() {
      this.isShowPhoto = false;
      this.isShowLocalPhoto = true;
      this.photoName = "";
      this.photoUrl_ = "";
    },
    async getPhoto() {
  let flag = 0; // 初始化标志变量
  const checkInterval = 1000; // 轮询间隔，单位为毫秒
  
  while (flag === 0) {
    try {
      const response = await fetch( this.$apiBaseUrl+`/stream_photo?name=${this.photoName}&style=2`);
      
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const url = response.url; // 获取响应的 URL
      
      // 发送请求来检查 url 指向的内容
      const contentResponse = await fetch(url); // 获取 url 指向的内容
      
      if (!contentResponse.ok) {
        throw new Error("Failed to fetch content from the URL");
      }

      const contentType = contentResponse.headers.get("Content-Type");

      if (contentType && contentType.includes("application/json")) {
        // 如果返回的是 JSON，则表示图片未准备好
        const jsonResponse = await contentResponse.json(); // 解析 JSON 响应
        console.log(jsonResponse); // 查看 JSON 内容
        // 检查 JSON 内容是否包含 success 字段并且值为 0
        if (jsonResponse.success === 0) {
          console.log("图片还在处理中...");
          flag = 0; // 图片未准备好，继续轮询
        }
      } else if (contentType && contentType.includes("image/jpeg")) {
        // 如果返回的是图像，则表示图片已准备好
        this.photoUrl = url; // 使用 URL 赋值给图片的 src
        console.log("图片已准备好，URL:", this.photoUrl);
        flag = 1; // 图片已准备好，设置标志位为 1
        this.isShowPhoto = true; // 显示图片
        this.isShowLocalPhoto = false; // 隐藏本地图片
      } else {
        console.log("未知的响应类型");
      }
    } catch (error) {
      console.error("获取照片时发生错误:", error);
    }

    // 等待指定的时间再进行下一次轮询
    await new Promise(resolve => setTimeout(resolve, checkInterval));
  }
},

    toggleDrawer() {
      this.drawerVisible = !this.drawerVisible;
    },
    handleBeforeUpload(file) {
      const supportedFormats = ["jpg", "jpeg", "png", "gif"];
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
          errorMessage += "不支持的图片格式。";
        }
        if (!isSizeValid) {
          errorMessage += "文件大小超过 100 MB。";
        }
        this.$message.error(errorMessage);
        return false;
      }
    },
    showConfirmDialog(file) {
      this.$confirm("确认上传此文件？", "提示", {
        confirmButtonText: "确定",
        cancelButtonText: "取消",
        type: "warning",
      })
        .then(() => {
          this.uploadFile(file);
        })
        .catch(() => {
          this.$message({
            type: "info",
            message: "已取消上传",
          });
        });
    },
    uploadFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    this.isShowPhoto = true;
    this.isShowLocalPhoto = true;
    this.photoUrl_ = e.target.result; // 将图片的 data URL 设置为 photoUrl
  };
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append("photo", file);

  this.$axios
    .post(this.uploadUrl, formData, {
      headers: {
        'Authorization': 'token5370ffbdcb08f3a7bfbf75287582539842d93c30', // 添加 Authorization 头
        'Content-Type': 'multipart/form-data' // 确保使用 multipart/form-data
      }
    })
    .then((response) => {
      this.handleSuccess(response, file);
    })
    .catch((error) => {
      this.handleError(error, file);
    });
},
    handleSuccess(response, file) {
      this.$message({
        type: "success",
        message: "上传成功!",
      });
      this.photoName = response.data.photoname;
    },
    handleError(error, file) {
      this.$message({
        type: "error",
      });
      console.error("上传失败:", error);
    },
    handleProgress(event, file) {
      console.log("上传进度:", event.percent);
    },
    onImageLoad() {
      console.log("图片加载成功");
    },
    onImageError() {
      console.error("图片加载失败");
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
  user-select: none;
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
  user-select: none;
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
video::-webkit-media-controls-play-button {
  display: none;
}
video::-webkit-media-controls-timeline {
  display: none;
}
</style>
