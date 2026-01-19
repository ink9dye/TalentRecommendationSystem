<template>
  <el-card :class="['talent-card', { active: isActive }]" shadow="hover">
    <div class="card-body">
      <div class="name-row">
        <strong>{{ talent.name }}</strong>
        <el-tag size="small" type="success">匹配度: {{ (talent.match_score * 100).toFixed(1) }}%</el-tag>
      </div>
      <div class="info-row">
        <span>{{ talent.institution }}</span>
      </div>
      <div class="metrics-row">
        <el-tag effect="plain" size="small">H-index: {{ talent.h_index }}</el-tag>
        <el-tag effect="plain" size="small">被引量: {{ talent.citations }}</el-tag>
      </div>
    </div>
  </el-card>
</template>

<script setup>
defineProps(['talent', 'isActive']);
</script>

<style scoped>
.talent-card { margin: 10px; cursor: pointer; border-left: 4px solid transparent; }
.talent-card.active { border-left-color: #409eff; background-color: #ecf5ff; }
.name-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.metrics-row { margin-top: 10px; display: flex; gap: 5px; }
</style>