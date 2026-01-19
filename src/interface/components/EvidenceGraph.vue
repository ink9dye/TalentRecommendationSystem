<template>
  <div ref="chartRef" style="width: 100%; height: 500px;"></div>
</template>

<script setup>
import { ref, onMounted, watch, onUnmounted } from 'vue';
// 1. 引入核心模块，提供绘图实例
import * as echarts from 'echarts/core';
// 2. 引入关系图 (Graph) 组件
import { GraphChart } from 'echarts/charts';
// 3. 引入提示框、标题、图例组件
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent
} from 'echarts/components';
// 4. 引入 Canvas 渲染器
import { CanvasRenderer } from 'echarts/renderers';

// 5. 必须手动注册这些组件
echarts.use([
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GraphChart,
  CanvasRenderer
]);

const props = defineProps(['data']);
const chartRef = ref(null);
let myChart = null;

// 初始化图表
onMounted(() => {
  if (chartRef.value) {
    myChart = echarts.init(chartRef.value);
    renderChart();
  }
  // 监听窗口缩放，防止图表变形
  window.addEventListener('resize', handleResize);
});

// 监听数据变化，当选择不同人才时更新图谱路径证据链
watch(() => props.data, () => {
  if (myChart) renderChart();
}, { deep: true });

const renderChart = () => {
  if (!props.data) return;

  const option = {
    title: {
      text: '推荐证据链 (知识图谱路径)',
      left: 'center',
      textStyle: { fontSize: 16, fontWeight: 'normal' }
    },
    tooltip: { trigger: 'item' },
    legend: { bottom: '0', left: 'center' },
    series: [{
      name: '证据链条',
      type: 'graph',
      layout: 'force', // 使用力导向布局展示异构网络
      data: props.data.nodes, // 包含“岗、词、作、人”等节点
      links: props.data.links, // 包含“技能需求、署名”等关系
      categories: [
        { name: '岗位' }, { name: '词汇' }, { name: '作品' }, { name: '作者' }
      ],
      label: { show: true, position: 'right' },
      force: {
        repulsion: 300, // 节点排斥力
        edgeLength: 100 // 连线长度
      },
      emphasis: { focus: 'adjacency' }, // 鼠标移入高亮关联路径
      roam: true, // 开启缩放和平移
      draggable: true // 开启节点拖拽
    }]
  };
  myChart.setOption(option);
};

const handleResize = () => myChart?.resize();

onUnmounted(() => {
  window.removeEventListener('resize', handleResize);
  myChart?.dispose(); // 销毁实例，释放内存
});
</script>