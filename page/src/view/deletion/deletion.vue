<template>
  <div style="width: 100%; height: 100%; background-color: white">
    <div style="font-size: 2vw; line-height: 12vh; font-weight: 900">
      🍈🍉缓存清理🍓🍇
    </div>
    <div>
      <el-alert
        style="width: 96%; margin-left: 2vw; height: 10vh"
        type="warning"
        :closable="false"
      >
        <div style="font-size: 1vw; font-weight: 900; text-align: left">
          温馨提示
        </div>
        <div style="font-size: 1vw; font-weight: 500">
          请留意各模块存储缓存大小，及时清理！
        </div>
      </el-alert>
      <div style="width: 96%; margin-left: 2vw">
        <div style="height: 1vh"></div>
        <el-divider content-position="left" style="font-size: 1.5vw">
          <div class="my_font">摄像头录像缓存</div>
        </el-divider>
        <div style="height: 1vh"></div>
        <div style="display: flex; height: 3vh">
          <div style="font-size: 0.8vw; line-height: 4vh; font-weight: 900">
            缓存大小:
          </div>
          <div style="width: 2%"></div>
          <el-input
            style="width: 8%; height: 3vh"
            placeholder="大小"
            v-model="inputRecord"
            :disabled="true"
          >
          </el-input>
          <span style="margin-left: 10px; font-size: 1.1vw; line-height: 5vh"
            >MB</span
          >
          <div style="width: 2%"></div>

          <el-button type="danger" class="my_button" @click="deleteRecord"
            >清理</el-button
          >
        </div>
        <div style="height: 1vh"></div>
        <div style="height: 1vh"></div>
        <el-divider content-position="left" style="font-size: 1.5vw">
          <div class="my_font">视频处理缓存</div>
        </el-divider>
        <div style="height: 1vh"></div>
        <div style="height: 1vh"></div>
        <div style="display: flex; height: 3vh">
          <div style="font-size: 0.8vw; line-height: 4vh; font-weight: 900">
            缓存大小:
          </div>
          <div style="width: 2%"></div>
          <el-input
            style="width: 8%; height: 3vh"
            placeholder="大小"
            v-model="inputVideo"
            :disabled="true"
          >
          </el-input>
          <span style="margin-left: 10px; font-size: 1.1vw; line-height: 5vh"
            >MB</span
          >
          <div style="width: 2%"></div>

          <el-button type="danger" class="my_button" @click="deleteVideo"
            >清理</el-button
          >
        </div>
        <div style="height: 1vh"></div>
        <div style="height: 1vh"></div>
        <el-divider content-position="left" style="font-size: 1.5vw">
          <div class="my_font">图像处理缓存</div>
        </el-divider>
        <div style="height: 1vh"></div>
        <div style="display: flex; height: 3vh">
          <div style="font-size: 0.8vw; line-height: 4vh; font-weight: 900">
            缓存大小:
          </div>
          <div style="width: 2%"></div>
          <el-input
            style="width: 8%; height: 3vh"
            placeholder="大小"
            v-model="inputPhoto"
            :disabled="true"
          >
          </el-input>
          <span style="margin-left: 10px; font-size: 1.1vw; line-height: 5vh"
            >MB</span
          >
          <div style="width: 2%"></div>
          <el-button type="danger" class="my_button" @click="deletePhoto"
            >清理</el-button
          >
        </div>
        <div style="height: 1vh"></div>
      </div>
    </div>
  </div>
  <!-- 以上为主内容 -->
</template>
<script>
export default {
  data() {
    return {
      inputRecord: "",
      inputVideo: "",
      inputPhoto: "",
    };
  },
  mounted() {
    this.recordStore();
    this.videoStore();
    this.photoStore();
  },
  methods: {
    async recordStore() {
      // 准备发送的数据
      let data = {
        type: "record",
      };

      try {
        // 使用 await 发送请求
        const response = await this.$axios.post(
          this.$apiBaseUrl + "/get_sizes",
          data
        );
        this.inputRecord = response.data.size;
      } catch (error) {
        // 捕获并处理错误
        console.error("错误:", error);
      }
    },

    async videoStore() {
      // 准备发送的数据
      let data = {
        type: "record",
      };

      try {
        // 使用 await 发送 POST 请求
        const response = await this.$axios.post(
          this.$apiBaseUrl + "/get_sizes",
          data
        );
        this.inputVideo = response.data.size;
      } catch (error) {
        // 捕获并处理错误
        console.error("错误:", error);
      }
    },

    async photoStore() {
      // 准备发送的数据
      let data = {
        type: "photo",
      };

      try {
        // 使用 await 发送 POST 请求
        const response = await this.$axios.post(
          this.$apiBaseUrl + "/get_sizes",
          data
        );
        this.inputPhoto = response.data.size;
      } catch (error) {
        // 捕获并处理错误
        console.error("错误:", error);
      }
    },

    deleteRecord() {
      let data = {
        type: "video",
      };
      this.$axios
        .post(this.$apiBaseUrl + "/delete", data)
        .then((response) => {
          this.recordStore();
        })
        .catch((error) => {
          console.error("错误:", error);
        });
    },
    deleteVideo() {
      let data = {
        type: "video",
      };
      this.$axios
        .post(this.$apiBaseUrl + "/delete", data)
        .then((response) => {
          this.videoStore();
        })
        .catch((error) => {
          console.error("错误:", error);
        });
    },
    deletePhoto() {
      let data = {
        type: "photo",
      };
      this.$axios
        .post(this.$apiBaseUrl + "/delete", data)
        .then((response) => {
          this.photoStore();
        })
        .catch((error) => {
          console.error("错误:", error);
        });
    },
  },
};
</script>
<style>
.my_font {
  font-size: 1.1vw;
  font-weight: 600;
}
.my_button {
  height: 5vh;
  width: 6vw;
  font-size: 0.8vw;
  margin-left: 0.5vw;
  margin-right: 0.5vw;
  font-weight: 900;
}
</style>
