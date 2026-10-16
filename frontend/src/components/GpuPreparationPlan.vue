<script setup lang="ts">
import type { GpuPreparation } from '../types'
defineProps<{ plan: GpuPreparation }>()
</script>
<template>
  <section class="gpu-preparation-plan" aria-label="GPU 运行准备方案">
    <strong>GPU 运行准备方案（尚未执行）</strong>
    <p>项目框架：{{ plan.framework || '待核验' }}；当前环境：{{ plan.runtime?.checked ? plan.runtime?.cuda_available ? '已枚举 GPU，运行前复检' : 'GPU 不可用' : '探测未完成' }}。</p>
    <p v-if="plan.preparation">{{ plan.preparation.reason }}</p>
    <p v-if="plan.gpu_environment_plan?.source_files?.length">源码兼容范围：{{ plan.gpu_environment_plan.source_files.join('、') }}</p>
    <p v-if="plan.gpu_environment_plan?.packages?.length">依赖方案：{{ plan.gpu_environment_plan.packages.join('、') }}</p>
    <details v-if="plan.gpu_environment_plan?.candidates?.length">
      <summary>查看已核验的兼容候选范围</summary>
      <p v-for="candidate in plan.gpu_environment_plan.candidates" :key="candidate.profile">{{ candidate.profile }}：{{ candidate.packages.join('、') }}</p>
    </details>
    <small>{{ plan.gpu_environment_plan?.warning || '只检查当前项目，不安装依赖、不修改源码、不启动训练。' }}</small>
    <p>流程：必要的源码修改 → 单独批准环境配置 → 依赖预检与安装 → 验证 GPU 运算 → 执行已批准的实验。安装或运算验证失败时停止，不退回 CPU。</p>
  </section>
</template>
<style scoped>
.gpu-preparation-plan { margin: 14px 0; padding: 14px; border: 1px solid #c8ded6; border-radius: 8px; background: #f1f8f5; overflow-wrap: anywhere; }
.gpu-preparation-plan p { margin: 9px 0; line-height: 1.6; }
.gpu-preparation-plan small { display: block; color: #627c73; line-height: 1.6; }
</style>
