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
            <i class="el-icon-camera"></i>
            <span slot="title">摄像头检测</span>        
        </el-menu-item>
        <el-menu-item index="/deletion">       
            <i class="el-icon-delete"></i>
            <span slot="title">缓存清理</span>          
        </el-menu-item>
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
      
      // 检查当前路由是否和目标路由相同，避免重复导航
      if (this.$route.path !== index) {
        // 导航到选定的路由
        this.$router.push(index).catch(err => {
          // 只捕获 NavigationDuplicated 错误，忽略其他错误
          if (err.name !== 'NavigationDuplicated') {
            console.error('Navigation Error:', err);
          }
        });
      } else {
        console.log('Already on the selected route:', index);
      }
    }

  }
}
</script>

<style>
.aside{
    
}
</style>