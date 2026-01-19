<template>
  <el-container
    class="home-container"
    v-loading="isLoading"
    :element-loading-text="loadingText"
    element-loading-background="rgba(255, 255, 255, 0.9)"
  >
    </el-container>
</template>

<script setup>
import { ref } from 'vue';
// 注意：删除了 SearchBar, TalentCard, EvidenceGraph 的手动 import
import { mockTalents, mockGraph } from '../mock/talentData';

// 如果加载动画没有样式，取消下面这行的注释
// import 'element-plus/es/components/loading/style/css';

const isLoading = ref(false);
const loadingText = ref('');
const talentList = ref([]);
const selectedTalent = ref(null);
const currentGraphData = ref(null);

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const handleSearch = async (query) => {
  isLoading.value = true;
  try {
    // 阶段 1：SBERT
    loadingText.value = '阶段 1/3: SBERT 语义向量化编码中...';
    await sleep(800);

    // 阶段 2：三路召回
    loadingText.value = '阶段 2/3: 三路(向量、标签、协同)并行召回中...';
    await sleep(1000);

    // 阶段 3：KGAT-AX 精排
    loadingText.value = '阶段 3/3: KGAT-AX 知识图谱全息特征重排序...';
    await sleep(800);

    talentList.value = mockTalents;
    if (talentList.value.length > 0) {
      selectTalent(talentList.value[0]);
    }
  } catch (error) {
    console.error('发现异常:', error);
  } finally {
    isLoading.value = false;
    loadingText.value = '';
  }
};

const selectTalent = (talent) => {
  selectedTalent.value = talent;
  currentGraphData.value = mockGraph;
};
</script>

<style scoped>
/* 保持原有样式 */
</style>