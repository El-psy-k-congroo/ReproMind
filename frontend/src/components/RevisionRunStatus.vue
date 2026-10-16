<script setup lang="ts">
import type { RevisionRunPlan } from '../types'
defineProps<{ plan: RevisionRunPlan | null; busy: boolean; stage: string; error: string }>()
</script>
<template>
  <div class="revision-run-status" aria-live="polite">
    <p v-if="stage" role="status">{{ stage }}</p>
    <template v-if="plan">
      <strong>{{ plan.action === 'prepare_code' ? '下一步：补全 GPU 兼容源码' : plan.action === 'approve_environment' ? '下一步：批准 GPU 环境准备与实验' : plan.action === 'blocked' ? '暂不能运行' : plan.action === 'already_created' ? '实验已经创建' : '运行条件已检查' }}</strong>
      <p>{{ plan.reason }}</p>
      <p v-if="plan.readiness?.gpu_environment_plan?.source_files?.length">需要检查：{{ plan.readiness.gpu_environment_plan.source_files.join('、') }}</p>
      <p v-if="plan.readiness?.gpu_environment_plan?.packages?.length">依赖方案：{{ plan.readiness.gpu_environment_plan.packages.join('、') }}</p>
      <small v-if="plan.readiness?.gpu_environment_plan?.warning">{{ plan.readiness.gpu_environment_plan.warning }}</small>
    </template>
    <p v-if="error && error !== plan?.reason" class="chat-error" role="alert">{{ error }}</p>
  </div>
</template>
