import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
      // 建议：自动生成类型声明文件
      dts: 'src/auto-imports.d.ts',
    }),
    Components({
      resolvers: [ElementPlusResolver()],
      // 建议：自动生成类型声明文件
      dts: 'src/components.d.ts',
    }),
  ],
  resolve: {
    alias: {
      // 必须补回，否则 @/ 开头的路径会失效
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
 optimizeDeps: {
    // 将重型依赖全部“预合并”成一个大的本地文件，减少浏览器请求次数
    include: [
      'vue',
      'element-plus/es',
      'echarts',
      '@vueuse/core'
    ]
  },
  server: {
    host: '127.0.0.1',
    // 强制不使用 HMR 覆盖层，减少浏览器和服务器之间的 WebSocket 通讯压力
    hmr: { overlay: false }
  }
})