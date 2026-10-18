<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { accountRequest, session } from '../session'

const route = useRoute(), router = useRouter()
const registering = computed(() => route.name === 'register')
const recovering = computed(() => route.name === 'recover')
const form = ref({ username: '', email: '', password: '', confirm: '', recovery_code: '', remember: false })
const visible = ref(false), busy = ref(false), error = ref(''), recoveryKey = ref(''), copied = ref(false)
const recovered = ref(false)
watch(() => route.name, () => { error.value = ''; form.value.password = ''; form.value.confirm = ''; visible.value = false })
function destination() {
  const next = typeof route.query.redirect === 'string' ? route.query.redirect : '/'
  return next.startsWith('/') && !next.startsWith('//') ? next : '/'
}
async function submit() {
  error.value = ''
  if ((registering.value || recovering.value) && form.value.password !== form.value.confirm) {
    error.value = '两次输入的密码不一致。'; return
  }
  busy.value = true
  try {
    if (recovering.value) {
      const body = await accountRequest('recover', { email: form.value.email, recovery_code: form.value.recovery_code, new_password: form.value.password })
      recoveryKey.value = body.recovery_code; recovered.value = true
    } else {
      const body = await accountRequest(registering.value ? 'register' : 'login', form.value)
      session.user = body.user; session.required = true; session.loaded = true
      if (registering.value) recoveryKey.value = body.recovery_code
      else await router.replace(destination())
    }
  } catch (err) { error.value = (err as Error).message }
  finally { busy.value = false }
}
async function copyKey() {
  try { await navigator.clipboard.writeText(recoveryKey.value); copied.value = true }
  catch { error.value = '请手动复制下方恢复密钥并妥善保存。' }
}
</script>

<template>
  <main class="auth-layout">
    <section class="auth-story">
      <a class="auth-brand" href="/"><span class="auth-logo">R</span><strong>ReproMind</strong><span>科研复现工作台</span></a>
      <div class="auth-story-body">
        <span class="auth-tag"><i></i> 从论文到实验的完整工作流</span>
        <h1>每一步复现，<br /><em>都有据可循。</em></h1>
        <p>把论文、作者代码、运行环境与实验结果放进同一个工作空间，和智能助手一起推进研究。</p>
        <div class="auth-workflow">
          <div><span>01</span><strong>读懂论文与代码</strong><small>聚合材料，定位方法与实验配置</small></div>
          <div><span>02</span><strong>构建环境与实验</strong><small>准备数据，运行、诊断与迭代</small></div>
          <div><span>03</span><strong>追踪结果与依据</strong><small>保留日志、参数、曲线和结论</small></div>
        </div>
      </div>
      <div class="auth-story-foot"><span>独立账号 · 私有项目空间</span><a href="https://github.com/El-psy-k-congroo/ReproMind" target="_blank" rel="noopener">开源与本地部署 ↗</a></div>
    </section>
    <section class="auth-form-side">
      <div v-if="recoveryKey" class="auth-form recovery-success">
        <div class="auth-success-mark">✓</div><h2>{{ recovered ? '密码已重置' : '你的工作空间已就绪' }}</h2>
        <p>请保存新的账号恢复密钥。忘记密码时可以用它重置密码；旧密钥已经失效。</p>
        <label class="field-label">账号恢复密钥<code class="recovery-key">{{ recoveryKey }}</code></label>
        <button class="button secondary full-width" @click="copyKey">{{ copied ? '已复制' : '复制恢复密钥' }}</button>
        <p class="field-hint">恢复密钥仅在这里完整展示。请保存在自己的密码管理器中，不要交给其他人。</p>
        <p v-if="error" class="auth-error" role="alert">{{ error }}</p>
        <button class="button primary full-width" @click="router.replace(recovered ? '/login' : destination())">{{ recovered ? '返回登录' : '已保存，进入工作台' }} <span>→</span></button>
      </div>
      <form v-else class="auth-form" @submit.prevent="submit">
        <span class="auth-form-kicker">你的研究，从这里继续</span>
        <h2>{{ recovering ? '找回账号密码' : registering ? '创建你的账号' : '登录科研工作台' }}</h2>
        <p>{{ recovering ? '使用注册时保存的恢复密钥设置新密码。' : registering ? '为论文、代码与实验建立独立的工作空间。' : '欢迎回来，继续推进你的复现实验。' }}</p>
        <div v-if="!recovering" class="auth-tabs" aria-label="账号操作">
          <RouterLink :to="{name:'login',query:route.query}" :class="{selected:!registering}">登录</RouterLink>
          <RouterLink :to="{name:'register',query:route.query}" :class="{selected:registering}">注册</RouterLink>
        </div>
        <label v-if="registering" class="field-label">昵称<input v-model="form.username" autocomplete="nickname" minlength="2" maxlength="40" required placeholder="你希望如何称呼自己" /></label>
        <label class="field-label">邮箱<input v-model="form.email" type="email" autocomplete="email" maxlength="128" required placeholder="name@example.com" /></label>
        <label v-if="recovering" class="field-label">账号恢复密钥<input v-model="form.recovery_code" autocomplete="off" required minlength="12" maxlength="160" placeholder="输入注册时保存的恢复密钥" /></label>
        <label class="field-label">{{ recovering ? '新密码' : '密码' }}
          <div class="password-control"><input v-model="form.password" :type="visible ? 'text' : 'password'" :autocomplete="registering || recovering ? 'new-password' : 'current-password'" :minlength="registering || recovering ? 10 : 1" maxlength="128" required :placeholder="registering || recovering ? '至少 10 位，包含字母和数字' : '输入你的密码'" /><button type="button" :aria-pressed="visible" :aria-label="visible ? '隐藏密码' : '显示密码'" @click="visible=!visible">{{ visible ? '隐藏' : '显示' }}</button></div>
        </label>
        <label v-if="registering || recovering" class="field-label">确认密码<input v-model="form.confirm" :type="visible ? 'text' : 'password'" autocomplete="new-password" minlength="10" maxlength="128" required placeholder="再次输入密码" /></label>
        <div v-if="!registering && !recovering" class="auth-options"><label><input v-model="form.remember" type="checkbox" /> 记住登录 30 天</label><RouterLink to="/recover">忘记密码？</RouterLink></div>
        <p v-if="error" class="auth-error" role="alert">{{ error }}</p>
        <button class="button primary full-width auth-submit" :disabled="busy">{{ busy ? '正在处理…' : recovering ? '重置密码' : registering ? '创建账号' : '登录' }}<span v-if="!busy">→</span></button>
        <p class="auth-privacy">{{ recovering ? '没有恢复密钥时，请联系部署维护者核实身份后处理。' : '每个账号的项目、材料、对话与实验记录相互隔离。' }}</p>
        <RouterLink v-if="recovering" class="auth-back" to="/login">← 返回登录</RouterLink>
      </form>
    </section>
  </main>
</template>
