import Vue from 'vue'
import Router from 'vue-router'
import Hello from '../view/hello/hello.vue'
import car_option1 from '../view/car/option1/car_option1.vue'
import car_option2 from '../view/car/option2/car_option2.vue'
import car_option3 from '../view/car/option3/car_option3.vue'
import people_option1 from '../view/people/option1/people_option1.vue'
import people_option2 from '../view/people/option2/people_option2.vue'
import people_option3 from '../view/people/option3/people_option3.vue'
Vue.use(Router)

export default new Router({
  routes: [
    {
      path:'/',
      name:'hello',
      component: Hello
    },
    {
      path:'/car/option1',
      name:'car_option1',
      component: car_option1
    },
    {
      path:'/car/option2',
      name:'car_option2',
      component: car_option2
    },
    {
      path:'/car/option3',
      name:'car_option3',
      component: car_option3
    },
    {
      path:'/people/option1',
      name:'people_option1',
      component: people_option1
    },
    {
      path:'/people/option2',
      name:'people_option2',
      component: people_option2
    },
    {
      path:'/people/option3',
      name:'people_option3',
      component: people_option3
    }
    
  ]
})
