// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router'
import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
import axios from 'axios' 

Vue.prototype.$axios = axios;
Vue.config.productionTip = false
Vue.use(ElementUI);
const apiBaseUrl = process.env.VUE_APP_API_BASE_URL;
console.log('API Base URL:', apiBaseUrl); // 这会输出你设置的值
Vue.prototype.$apiBaseUrl = apiBaseUrl;
/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
