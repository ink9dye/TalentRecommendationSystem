// E:\PythonProject\TalentRecommendationSystem\src\main.ts

import { createApp } from 'vue'
// 注意路径：从当前目录进入 interface 文件夹找 App.vue
import App from './interface/App.vue'

import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

const app = createApp(App)
app.use(ElementPlus)
app.mount('#app')