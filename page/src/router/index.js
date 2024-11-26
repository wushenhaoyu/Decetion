import Vue from 'vue'
import Router from 'vue-router'
import Hello from '../view/hello/hello.vue'
import camera from '../view/detection/option1/camera.vue'
import video from '../view/detection/option2/video.vue'
import photo from '../view/detection/option3/photo.vue'
import deletion from '../view/deletion/deletion.vue'
Vue.use(Router)

export default new Router({
  routes: [
    {
      path:'/',
      name:'hello',
      component: Hello,
      meta: {
        keepAlive: true // 需要被缓存
      }
    },
    {
      path:'/detection/option1',
      name:'camera',
      component: camera,
      meta: {
        keepAlive: true // 需要被缓存
      }

    },
    {
      path:'/detection/option2',
      name:'video',
      component: video,
      meta: {
        keepAlive: true // 需要被缓存
      }
    },
    {
      path:'/detection/option3',
      name:'photo',
      component: photo,
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
