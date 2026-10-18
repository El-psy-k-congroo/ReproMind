<script setup lang="ts">
import { ref } from 'vue'
import { accountRequest, session, logout } from '../session'
const current = ref(''), password = ref(''), confirm = ref(''), recoveryKey = ref(''), error = ref(''), busy = ref(false), copied = ref(false)
async function changePassword() {
  error.value = ''; recoveryKey.value = ''
  if (password.value !== confirm.value) { error.value = '两次输入的密码不一致。'; return }
  busy.value = true
  try {
    const result = await accountRequest('password', {current_password:current.value,new_password:password.value})
    recoveryKey.value = result.recovery_code; current.value = ''; password.value = ''; confirm.value = ''
  } catch (err) { error.value = (err as Error).message }
  finally { busy.value = false }
}
async function copyKey() { try { await navigator.clipboard.writeText(recoveryKey.value); copied.value = true } catch { error.value = '请手动复制恢复密钥。' } }
async function signOut() { try { await logout() } catch (err) { error.value = (err as Error).message } }
</script>
<template>
  <section class="account-page page-enter">
    <header class="topbar"><div><span class="kicker">ACCOUNT</span><h1>账号与安全</h1><p>管理登录信息和账号恢复方式。</p></div></header>
    <div class="account-grid">
      <article class="account-card"><div class="account-avatar">{{ session.user?.username.slice(0,1).toUpperCase() }}</div><h2>{{ session.user?.username }}</h2><p>{{ session.user?.email }}</p><div class="account-detail"><span>工作空间</span><strong>个人项目空间</strong></div><div class="account-detail"><span>数据访问</span><strong>仅限本人账号</strong></div><button class="button secondary full-width" @click="signOut">退出登录</button></article>
      <form class="account-card" @submit.prevent="changePassword"><h2>修改密码</h2><p>修改后其他设备的登录会话将失效。</p><label class="field-label">当前密码<input v-model="current" type="password" autocomplete="current-password" required /></label><label class="field-label">新密码<input v-model="password" type="password" autocomplete="new-password" required minlength="10" maxlength="128" placeholder="至少 10 位，包含字母和数字" /></label><label class="field-label">确认新密码<input v-model="confirm" type="password" autocomplete="new-password" required minlength="10" maxlength="128" /></label><p v-if="error" class="auth-error" role="alert">{{ error }}</p><div v-if="recoveryKey" class="account-success" role="status"><strong>密码已更新，请保存新的恢复密钥。</strong><code class="recovery-key">{{ recoveryKey }}</code><button type="button" class="button secondary" @click="copyKey">{{ copied ? '已复制' : '复制密钥' }}</button></div><button class="button primary" :disabled="busy">{{ busy ? '正在保存…' : '更新密码' }}</button></form>
    </div>
  </section>
</template>
