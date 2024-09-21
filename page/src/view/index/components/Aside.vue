<template>
    <div>
      <el-menu 
        default-active="/" 
        class="el-menu-vertical-demo" 
        @open="handleOpen" 
        @close="handleClose" 
        @select="handleSelect"
        :collapse="isCollapse">
        <el-menu-item index="/">
          <i class="el-icon-s-home"></i>
          <span slot="title">起始页</span>
        </el-menu-item>
        <el-submenu index="camera">
          <template slot="title">
            <i class="el-icon-video-camera"></i>
            <span slot="title">摄像头检测</span>
          </template>
          </el-submenu>
          <el-submenu index="video">
          <template slot="title">
            <i class="el-icon-video-camera"></i>
            <span slot="title">视频检测</span>
          </template>
          </el-submenu>
          <el-submenu index="photo">
          <template slot="title">
            <i class="el-icon-video-camera"></i>
            <span slot="title">图片检测</span>
          </template>
          </el-submenu>
      </el-menu>
    </div>
  </template>


<script>
export default {
  data() {
    return {
      isCollapse: true
    };
  },
  methods: {
    handleOpen(key, keyPath) {
      console.log('Open:', key, keyPath);
    },
    handleClose(key, keyPath) {
      console.log('Close:', key, keyPath);
    },
    handleSelect(index) {
      console.log('Selected index:', index);
      // Navigate to the selected route
      this.$router.push(index).catch(err => {
        // Handle navigation errors, if any
        console.error('Navigation Error:', err);
      });
    }
  }
}
</script>

<style>
.aside{
    
}
</style>