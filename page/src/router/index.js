import Vue from 'vue'
import Router from 'vue-router'
import camera from '../view/detection/option1/camera.vue'
import deletion from '../view/deletion/deletion.vue'
Vue.use(Router)

export default new Router({
  routes: [
    {
      path:'/',
      name:'camera',
      component: camera,
      meta: {
        keepAlive: true // 需要被缓存
      }

    },
    {
      path:'/deletion',
      name:'deletion',
      component:deletion,
      meta: {
        keepAlive: true // 需要被缓存
      }
    },
    
    
  ]
})
