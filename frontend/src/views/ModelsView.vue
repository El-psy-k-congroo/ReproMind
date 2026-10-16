<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api } from '../api'
import type { ModelCatalog, ModelRouting } from '../types'
const state = ref<ModelRouting | null>(null)
const catalog = ref<ModelCatalog | null>(null)
const choices = ref<Record<string, string>>({})
const busy = ref(false), error = ref(''), notice = ref('')
async function load() {
  error.value = ''; busy.value = true
  try {
    state.value = await api.modelRouting()
    choices.value = Object.fromEntries(state.value.groups.map(g => [g.id, g.model]))
    catalog.value = await api.listModels()
  } catch (err) { error.value = (err as Error).message }
  finally { busy.value = false }
}
async function save() {
  if (!state.value || busy.value) return
  busy.value = true; error.value = ''; notice.value = ''
  try {
    state.value = await api.saveModelRouting({ ...choices.value }, state.value.revision)
    choices.value = Object.fromEntries(state.value.groups.map(g => [g.id, g.model]))
    notice.value = '已保存。后续模型调用按此配置路由；正在执行的模型请求和历史记录不改写。'
  } catch (err) { error.value = (err as Error).message }
  finally { busy.value = false }
}
onMounted(load)
</script>
<template>
  <section class="model-routing-page">
    <header><span class="eyebrow">MODEL DISPATCH</span><h1>多模型调度</h1><p>调度器按能力组为 Agent 选择模型，不必逐个配置。未选择时保留当前默认模型。</p></header>
    <p v-if="error" class="chat-error" role="alert">{{ error }}</p><p v-if="notice" role="status">{{ notice }}</p>
    <div class="model-routing-grid">
      <article v-for="group in state?.groups || []" :key="group.id" class="model-routing-card">
        <span class="eyebrow">{{ group.id }}</span><h2>{{ group.label }}</h2>
        <p>{{ group.agents.join(' · ') }}</p>
        <label :for="`routing-${group.id}`">共用模型</label>
        <select :id="`routing-${group.id}`" v-model="choices[group.id]" :disabled="busy" :aria-label="`${group.label}模型`">
          <option value="">沿用当前默认配置</option>
          <option v-if="group.model && !catalog?.models.some(m => m.id === group.model)" :value="group.model">{{ group.model }}（目录暂未确认）</option>
          <option v-for="model in catalog?.models || []" :key="model.id" :value="model.id">{{ model.label }}</option>
        </select>
        <small>当前生效：{{ group.effective_model }}</small>
      </article>
    </div>
    <div class="model-routing-footer"><p>{{ state?.note }}</p><p>配置只保存模型名称，不保存密钥。普通训练、安装、数据核对等确定性步骤不冒充模型调用。对话和代码工作台可分别显式覆盖本次模型。</p><button class="button primary" :disabled="busy || !state" @click="save">{{ busy ? '正在处理…' : '保存调度配置' }}</button><button class="button secondary" :disabled="busy" @click="load">重新读取</button></div>
  </section>
</template>
