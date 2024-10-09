<template>
  <div style="height: 100%; width: 100%; display: flex; justify-content: center; align-items: center;">
    <img :src="imageSrc" alt="Video Image" />
  </div>
</template>


<script>
export default {
  data() {
    return {
      imageSrc: ''
    };
  },
  created() {
    this.initWebSocket();
    console.log('你好')
  },
  beforeDestroy() {
    this.closeWebSocket();
  },
  methods: {
    initWebSocket() {
      const wsUrl = `ws://127.0.0.1:8000/ws/video/wms/`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('WebSocket connection opened');
      };

      this.ws.onmessage = (evt) => {
        try {
          const vData = JSON.parse(evt.data);
          this.imageSrc = vData.message;
          
        } catch (error) {
          console.error('Error parsing message:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket connection closed');
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    },
    closeWebSocket() {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.close();
      }
    }
  }
};
</script>

<style>
/* 可以在这里添加样式 */
</style>

