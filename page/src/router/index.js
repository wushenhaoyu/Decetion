import Vue from 'vue'
import Router from 'vue-router'
import Hello from '../view/hello/hello.vue'
import camera from '../view/detection/option1/camera.vue'
import video from '../view/detection/option2/video.vue'
import photo from '../view/detection/option3/photo.vue'
Vue.use(Router)

export default new Router({
  routes: [
    {
      path:'/',
      name:'hello',
      component: Hello
    },
    {
      path:'/detection/option1',
      name:'camera',
      component: camera
    },
    {
      path:'/detection/option2',
      name:'video',
      component: video
    },
    {
      path:'/detection/option3',
      name:'photo',
      component: photo
    },
    
    
  ]
})
