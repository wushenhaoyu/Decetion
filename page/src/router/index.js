import Vue from 'vue'
import Router from 'vue-router'
import Hello from '../view/hello/hello.vue'
import car_option1 from '../view/detection/option1/camera_option1.vue'
import car_option2 from '../view/detection/option2/video_option2.vue'
import car_option3 from '../view/detection/option3/photo_option3.vue'
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
      name:'camera_option1',
      component: car_option1
    },
    {
      path:'/detection/option2',
      name:'video_option2',
      component: car_option2
    },
    {
      path:'/detection/option3',
      name:'photo_option3',
      component: car_option3
    },
    
    
  ]
})
