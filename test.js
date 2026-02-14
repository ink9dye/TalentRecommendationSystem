// test-esm.js - 使用 ES 模块语法
import fs from 'fs';
import { performance } from 'perf_hooks';

console.log('=== ES模块测试 ===');

// 测试1: CPU性能
console.time('cpu-test');
let sum = 0;
for (let i = 0; i < 1000000; i++) {
  sum += i;
}
console.timeEnd('cpu-test');

// 测试2: 文件读取性能
if (fs.existsSync('./package.json')) {
  console.time('file-read');
  const data = fs.readFileSync('./package.json', 'utf8');
  console.timeEnd('file-read');
  console.log(`文件大小: ${data.length} 字节`);
} else {
  console.log('找不到 package.json');
}

console.log('=== 测试结束 ===');