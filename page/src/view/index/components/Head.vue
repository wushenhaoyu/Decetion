<template>
    <div class="head">
      <el-menu :default-active="activeIndex" class="el-menu-demo head" mode="horizontal" @select="handleSelect">
        <div style="height: 6vh;">
            <a :href="'https://www.paddlepaddle.org.cn/'" target="_blank" rel="noopener noreferrer">
            <img src="../../../assets/logo.png" alt="icon" height="100%" />
            </a>
        </div>
        <div class="spacer"></div>
        <el-menu-item index="1">AIStudio部署版</el-menu-item>
      </el-menu>
    </div>
  </template>
  
<script>
  export default {
    data() {
      return {
        activeIndex: '1',
        activeIndex2: '1'
      };
    },
    methods: {
      handleSelect(key, keyPath) {
        console.log(key, keyPath);
      }
    }
  }
</script>

<style >
.head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.el-menu-demo {
  display: flex;
  flex-grow: 1;
}

.spacer {
  flex-grow: 1;
}

.el-menu-item {
  line-height: 60px;
}
</style>