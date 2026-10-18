import { reactive } from 'vue'

export type AccountUser = { id: string; username: string; email: string; role: string }
export const session = reactive({ loaded: false, required: false, user: null as AccountUser | null })
export const startup = reactive({ error: '' })
const BASE = import.meta.env.VITE_API_BASE_URL || '/api'

export async function accountRequest(path: string, payload?: unknown) {
  const response = await fetch(`${BASE}/auth/${path}`, {
    method: payload === undefined ? 'GET' : 'POST', credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json', 'X-ReproMind-Request': '1' },
    body: payload === undefined ? undefined : JSON.stringify(payload),
    signal: AbortSignal.timeout(20000),
  })
  const body = await response.json()
  if (!response.ok) {
    const detail = body.detail
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail)
      ? detail.map(item => item.msg).join('；') : '请求失败，请稍后重试')
  }
  return body
}

let pending: Promise<void> | null = null
export async function loadSession(force = false) {
  if (session.loaded && !force) return
  if (!pending) pending = (async () => {
    const body = await accountRequest('session')
    session.required = body.auth_required
    session.user = body.user
    session.loaded = true
  })().finally(() => { pending = null })
  return pending
}

export async function logout() {
  await accountRequest('logout', {})
  session.user = null
  // Clear in-memory project and conversation state across account switches.
  window.location.assign('/login')
}
