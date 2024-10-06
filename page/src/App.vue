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
  },created() {
    this.setFavicon('./assets/logo.png'); // 设置图标的路径
  },
  methods: {
    setFavicon(iconUrl) {
      let link = document.querySelector("link[rel~='icon']");
      if (!link) {
        link = document.createElement('link');
        link.rel = 'icon';
        document.getElementsByTagName('head')[0].appendChild(link);
      }
      link.href = iconUrl;
    }
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
