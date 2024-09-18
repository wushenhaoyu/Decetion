<template>
  <div id="app">
    <index></index>
  </div>
</template>

<script>
import index from './view/index/index.vue'  // 确保文件扩展名是 .vue
export default {
  name: 'App',
  components: {
    index
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}
</style>
