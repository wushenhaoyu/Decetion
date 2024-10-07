<template>
<div style="width: 100%; height: 100%" >
    <!-- 固定在右边的抽屉 -->
    <div :class="drawer_class_ctrl" style="width: 15vw">
      <div class="drawer-content" >
        <div class="right-log-head" style="line-height: 6vh;position: absolute;z-index: 3  ;height: 6vh;">处理中心</div>
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
    
    <div v-if="isShowCameraSelectList" style="position: absolute;height: 90%;width: 90%;z-index: 4" @click="closeSelectCamera"></div>
    <div v-if="isShowRecordList" style="position: absolute;height: 90%;width: 90%;z-index: 4" @click="controlRecordList"></div>
    <!-- 以下为主内容 -->
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
    <div style="height: 5%"></div>
    <div class="main">
      <div style="height: 100%; width: 70%">
        <div style="height: 80%; width: 100%;position: relative;" class="upload">
          <!-- 显示视频  -->
          <div align="center"><img  :src="cameraUrl" id="video" style="height:100%;">
          </div>
          <!-- 显示视频  -->
           
          <!--提示部分-->
          <div v-if="isShowCameraSelectList" style="height: 80%;width: 60%;position: absolute;left: 50%;top: 50%;transform: translate(-50%,-50%);z-index: 8;background-color: white;border-radius: 0.5vw;">
            <div style="width: 100%; height: 7vh;border-top-left-radius: 0.5vw;border-top-right-radius: 0.5vw;background-color: rgb(66,159,255);text-align: center;line-height: 7vh;;font-size: 2vw;color: white;font-weight: 900;">选择摄像头</div>
            <div v-if="cameraList.length>0">
                <div
                  v-for="(item, index) in cameraList"
                  :key="index"
                  style="height: 10vh;line-height: 10vh;text-align: center;font-size: 2vw;display: flex;justify-content: space-between;"
                >
                  <div style="width: 70%;">{{ item[1] }}</div>
                  <div style="width: 30%;"><el-button type="primary" style="font-size: 2vw;" @click="selectCamera(item[0])">选择</el-button></div>
                </div>
            </div>
            <div v-else style="text-align: center;line-height: 40vh;font-size: 3vw;color: gray;font-weight: 900;">
              暂无摄像头设备
            </div>
            
          </div>
          <!--提示部分-->
          
          <!--获取录制视频界面-->
          <div v-if="isShowRecordList" style="height: 80%;width: 60%;position: absolute;left: 50%;top: 50%;transform: translate(-50%,-50%);z-index: 8;background-color: white;border-radius: 0.5vw;">
            <div style="width: 100%; height: 7vh;border-top-left-radius: 0.5vw;border-top-right-radius: 0.5vw;background-color: rgb(66,159,255);text-align: center;line-height: 7vh;;font-size: 2vw;color: white;font-weight: 900;">录制视频资源</div>
            <div v-if="recordList.length>0" style="overflow-y: scroll;height: 40vh">
                <div
                  v-for="(item, index) in recordList"
                  :key="index"
                  style="height: 10vh;line-height: 10vh;text-align: center;font-size: 2vw;display: flex;justify-content: space-between;"
                >
                  <div style="width: 70%;">{{ item }}</div>
                  <div style="width: 30%;"><el-button type="primary" style="font-size: 2vw;" @click="selectRecord(item)">导出</el-button></div>
                </div>
            </div>
            <div v-else style="text-align: center;line-height: 40vh;font-size: 3vw;color: gray;font-weight: 900;">
              暂无录制视频资源
            </div>
            
          </div>
          <!--获取录制视频界面-->
        </div>
        <div style="height: 5%"></div>
        <!-- 视频下方操作按钮-->
        <div
          class="bottom-ctrl"
          style="height: 15%; width: 100%; font-size: 2vw"
        >
          <div class="bottom-ctrl-one">
            <el-button
              type="primary"
              class="bottom-button"
              @click="getCamera"
              >{{ isShowCamera ? "关闭摄像头" : "开启摄像头" }}</el-button
            >
            <el-button type="primary" class="bottom-button" @click="controlRecord">{{ isStartRecord ? "关闭录制" :"开启录制" }}</el-button>
            <el-button type="primary" class="bottom-button" >执行检测</el-button>
            <el-button type="primary" class="bottom-button" @click="controlRecordList">导出视频</el-button
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
import myvideo from "../../video/myvideo.vue";
export default {
  components: {
    myvideo,
  },
  data() {
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
      people_detector_enable: false, // 行人监测
      people_tracker_enable: false,
      people_attribute_enable: false,
      vehicle_detector_enable: false,//车辆监测
      vehicle_tracker_enable:false,
      vehicle_press_detector_enable: false,
      vehicle_license_enable: false,
      vehicle_attribute_enable: false,
      vehicle_invasion_enable:false,
      isShowCamera: false,
      drawerVisible: false,
      activeIndex: "4", // 更新为菜单项的实际索引
      cameraUrl: "",
      uploadUrl: "",
      showProgress: false,
      progressPercentage: 0,
      cameraList :[],
      isShowCameraSelectList:false,
      isStartRecord:false,
      isShowRecordList:false,
      recordList:[],
      detailPhotoName: "",
      logDetail: false,
    };
  },
    created() {
    // 在生命周期钩子里初始化 cameraUrl
    this.cameraUrl = '/api/' + "livedisplay";
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
            '/api/' + `stream_photo?name=${this.detailPhotoName}&style=3`
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
            '/api/' + `stream_video?name=${this.detailPhotoName}&style=4`
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
    
    handleCurrentChange(page) {
      this.currentPage = page;
    },
    
    startRecordVideo(){
      this.$axios.post('/api/' + 'livedisplayRecord').then(res => {
      })
    },
    async getLog() {
      try {
        const response = await this.$axios.post('/api/'+"log");
        console.log(response);
        if (response.status === 200) {
          const convertedPeopleLog = response.data.people_log.map((item) => {
            return {
              id: "行人",
              time: item[0],
              location: `(${Math.floor(item[1][0])}, ${Math.floor(
                item[1][1]
              )}, ${Math.floor(item[1][2])}, ${Math.floor(item[1][3])})`,
              name: item[2],
            };
          });

          // 转换 vehicle_log
          const convertedVehicleLog = response.data.vehicle_log.map((item) => {
            return {
              id: "车辆",
              time: item[0],
              location: `(${Math.floor(item[1][0])}, ${Math.floor(
                item[1][1]
              )}, ${Math.floor(item[1][2])}, ${Math.floor(item[1][3])})`,
              name: item[2],
            };
          });

          // 合并结果
          const combinedLogs = [...convertedPeopleLog, ...convertedVehicleLog];

          // 将合并后的数据赋值给 paginatedData
          this.tableData = this.tableData.concat(combinedLogs);
          this.total = this.tableData.length;
          console.log(this.tableData);
        }
      } catch (error) {
        console.error(
          "请求失败:",
          error.response ? error.response.data : error
        );
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
      this.$axios.post('/api/'+'ConfirmParams', data).then(res => {
      })
    },
    checkParameter(value){
      var that = this
      if (value){ //有东西开启了，要保证额外功能开启的时候，保证追踪或者检测开启
        const people_list = [that.people_attribute_enable]
        for (let i = 0; i < people_list.length; i++) {
          if(people_list[i]){
            that.people_tracker_enable = true
          }
        }
        const vehicle_list = [that.vehicle_attribute_enable, that.vehicle_license_enable, that.vehicle_press_detector_enable,that.vehicle_invasion_enable]
        for (let i = 0; i < vehicle_list.length; i++) {
          if(vehicle_list[i]){
            that.vehicle_tracker_enable = true
          }
        }
      }else{ //检测关闭，额外功能也要关闭
        if (!(that.people_detector_enable && that.people_tracker_enable))
        {
            that.people_attribute_enable = false
        }
        if (!(that.vehicle_detector_enable && that.vehicle_tracker_enable))
        {
            that.vehicle_attribute_enable = false
            that.vehicle_license_enable = false
            that.vehicle_press_detector_enable = false
            that.vehicle_invasion_enable = false
        }

      }
    },
    selectCamera(id){
      let data = {
        camId : id
      }
      this.$axios.post('/api/'+"Camchoice",data).then((response) => {
          console.log(response)
          if (response.data.success == 1)
          {
            this.isShowCameraSelectList = false
            this.switchCamera()
          }
        })
    },
    closeSelectCamera(){
      this.isShowCameraSelectList = false
    },
    controlRecordList(){
        if (this.isShowRecordList){
          this.hideRecordList()
        }else{
          this.showRecordList()
        }
    },
    getRecordList(){
      this.$axios.get('/api/'+"getAllRecordFile").then((response) => {
          this.recordList = response.data.files
        })
    },
    showRecordList(){
      this.isShowRecordList = true
      this.getRecordList()
    },
    hideRecordList(){
      this.isShowRecordList = false
    },
    controlRecord(){
      if(this.isStartRecord){
        this.closeRecord()
      }else{
        this.startRecord()
      }
    },
    startRecord(){
      this.$axios.post('/api/'+"video_record_on").then((response) => {
          if(response.data.status){
            this.$message({
            type: 'success',
            message: '开始录制'
          });
          this.isStartRecord = true;
          }else{
            this.$message({
            type: 'error',
            message: '开始录制失败，请检查摄像头'
          });
          this.isStartRecord = false;
          }
        })
    },
    closeRecord(){
      this.$axios.post('/api/'+"video_record_off").then((response) => {
          if(response.data.status){
            this.$message({
            type: 'success',
            message: '关闭录制'
          });
          this.isStartRecord = false;
          }else{
            this.$message({
            type: 'error',
            message: '关闭录制失败，请检查摄像头'
            })
          }
      })
    },
    async selectRecord(value) {
      let data = {
        name: value
      };

      try {
        const response = await this.$axios.get('/api/'+'stream_record_download', {
          params: data,
          responseType: 'blob' // 以 Blob 形式接收响应
        });

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'record_file'); // 设置下载文件名
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link); // 清理临时元素
        window.URL.revokeObjectURL(url); // 释放 URL 对象
      } catch (error) {
        console.error('Error downloading file:', error);
      }
    },
    getCamera(){
      if (this.isShowCamera)
      {
        this.switchCamera()
      }else{
        this.$axios.get('/api/'+"getAllCam").then((response) => {
            if (response.data.success == 1)
            {
              this.cameraList = response.data.cam  
            }
          })
          this.isShowCameraSelectList = true
      }

    },
    switchCamera() {
      if (this.isShowCamera) {
        this.isShowCamera = false;
        clearInterval(this.logPollingInterval);
        this.$axios.get('/api/'+"closecam").then((response) => {
            //src让其为空
          })
          this.cameraUrl = ""
      } else {
        this.isShowCamera = true;
        this.sendParameters()
        this.$axios.get('/api/'+"opencam").then((response) => {
          
          })
          console.log("abc");
        this.cameraUrl = '/api/'+"livedisplay"
        console.log("abc");
        this.startLogPolling();
        console.log("abc");
      }
    },
    startLogPolling() {
      // 每隔 1000 毫秒（1 秒）调用 getLog
      console.log("abc");
      
      this.logPollingInterval = setInterval(async () => {
        await this.getLog();
      }, 1000);
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
      this.showProgress = false;
      this.progressPercentage = 0;
      this.$message.success("上传成功");
    },
    handleError(error, file, fileList) {
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
  user-select:none;
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
  user-select:none;
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
  border-radius: 8px; /* 可选: 让边角变圆 */
  margin-bottom: -2vh;
  background-color: black;
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
.drawer_switch{
  height: 8vh;
  
}
.drawer-open {
  position: fixed;
  top: 10vh;
  right: 0;
  width: 15vw;
  height: 100%;
  background-color: #fff;
  box-shadow: -1px 0 2px rgba(0, 0, 0, 0.5);
  z-index: 3;
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
  z-index: 3;
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
video::-webkit-media-controls-play-button {
  display: none;
}
video::-webkit-media-controls-timeline {
  display: none;
}
</style>
