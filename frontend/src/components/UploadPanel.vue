<script setup lang="ts">
import { ref, watch } from 'vue'
import type { ArtifactKind } from '../types'

const props = defineProps<{ busy?: boolean; initialKind?: ArtifactKind }>()
const emit = defineEmits<{ upload: [kind: ArtifactKind, file: File] }>()
const kind = ref<ArtifactKind>(props.initialKind || 'paper')
watch(() => props.initialKind, value => { if (value) kind.value = value })
const fileInput = ref<HTMLInputElement | null>(null)
const dragging = ref(false)

const labels: Record<ArtifactKind, string> = {
  paper: '论文', code: '代码', config: '配置', result: '结果', note: '补充说明', dataset: '数据集',
}

function send(files: FileList | null) {
  const file = files?.[0]
  if (file) emit('upload', kind.value, file)
  if (fileInput.value) fileInput.value.value = ''
}

function drop(event: DragEvent) {
  dragging.value = false
  send(event.dataTransfer?.files || null)
}
</script>

<template>
  <section class="upload-panel">
    <div class="section-title-row">
      <div><span class="kicker">ADD EVIDENCE</span><h2>上传证据</h2></div>
      <select v-model="kind" aria-label="证据类型">
        <option v-for="(label, value) in labels" :key="value" :value="value">{{ label }}</option>
      </select>
    </div>
    <button
      class="drop-zone"
      :class="{ dragging }"
      :disabled="props.busy"
      @click="fileInput?.click()"
      @dragover.prevent="dragging = true"
      @dragleave.prevent="dragging = false"
      @drop.prevent="drop"
    >
      <span class="upload-icon">↑</span>
      <strong>{{ props.busy ? '正在解析…' : `拖入或选择${labels[kind]}文件` }}</strong>
      <small>{{ kind === 'dataset' ? '上传作者数据 ZIP / TAR.GZ 或 ind.* 文件；当前单文件上限 40 MB' : 'PDF / ZIP / YAML / JSON / CSV / 源代码，单文件不超过 40 MB' }}</small>
    </button>
    <input ref="fileInput" hidden type="file" @change="send(($event.target as HTMLInputElement).files)" />
  </section>
</template>
