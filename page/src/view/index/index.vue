<template>
  <el-container>
    <el-header style="padding: 0;"><Head></Head></el-header>
  <el-container>
     <Aside></Aside>
    <el-main><keep-alive><router-view></router-view></keep-alive></el-main>
  </el-container>
</el-container>
</template>

<script>
import Aside from './components/Aside'
import Head from './components/Head.vue';
export default {
    components: {
    Aside,
    Head
  }
}
</script>

<style>
 .el-header, .el-footer {
    background-color: #B3C0D1;
    color: #333;
    text-align: center;
    line-height: 60px;
    height: 10vh;
  }
  
  .el-aside {
    background-color: #D3DCE6;
    color: #333;
    text-align: center;
    line-height: 200px;
  }
  
  .el-main {
    background-color: #E9EEF3;
    color: #333;
    text-align: center;
    height: 90vh;
  }
  
  body > .el-container {
    margin-bottom: 40px;
  }
  
  .el-container:nth-child(5) .el-aside,
  .el-container:nth-child(6) .el-aside {
    line-height: 260px;
  }
  
  .el-container:nth-child(7) .el-aside {
    line-height: 320px;
  }

</style>