<template>
  <div class="search-box">
    <el-input
      v-model="query"
      placeholder="输入岗位描述或科研需求"
      class="input-with-select"
      @keyup.enter="onSearch"
    >
      <template #append>
        <el-button @click="onSearch">召回并精排</el-button>
      </template>
    </el-input>
  </div>
</template>

<script setup>
import { ref } from 'vue';
const query = ref('');
const emit = defineEmits(['search']);
const onSearch = () => emit('search', query.value);
</script>

<style scoped>
.search-box { width: 500px; }
</style>