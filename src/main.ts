// src/main.ts
import { createApp } from 'vue'
import App from './interface/App.vue'

const app = createApp(App)

// 注意：这里删除了 ElementPlus 的全局引入和 app.use(ElementPlus)
// 删除了 'element-plus/dist/index.css' 的引入

app.mount('#app')