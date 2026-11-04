import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import DashboardView from './views/DashboardView.vue'
import ProjectView from './views/ProjectView.vue'
import InsightsView from './views/InsightsView.vue'
import './styles.css'
import { loadSession, session, startup } from './session'
import { recoverRouteImport } from './routeRecovery'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: () => import('./views/AuthView.vue'), meta: {public:true} },
    { path: '/register', name: 'register', component: () => import('./views/AuthView.vue'), meta: {public:true} },
    { path: '/recover', name: 'recover', component: () => import('./views/AuthView.vue'), meta: {public:true} },
    { path: '/account', name: 'account', component: () => import('./views/AccountView.vue') },
    { path: '/', name: 'dashboard', component: DashboardView },
    { path: '/projects/:id', name: 'project', component: ProjectView },
    { path: '/projects/:id/demo', name: 'project-demo', component: () => import('./views/DemoView.vue') },
    { path: '/insights', name: 'insights', component: InsightsView },
    { path: '/agents', name: 'agents', component: () => import('./views/AgentsView.vue') },
    { path: '/models', name: 'models', component: () => import('./views/ModelsView.vue') },
  ],
})

router.onError((error, to) => {
  // Access sessionStorage lazily: browser privacy settings may deny it.
  const storage = {
    getItem: (key: string) => window.sessionStorage.getItem(key),
    setItem: (key: string, value: string) => window.sessionStorage.setItem(key, value),
  }
  const result = recoverRouteImport(error, to.fullPath, storage, target => window.location.assign(target))
  if (result === 'failed') startup.error = '页面资源加载失败，请检查服务后点击“重新连接”。'
  if (result === 'unrelated') startup.error = '页面打开失败，请点击“重新连接”后重试。'
})

router.beforeEach(async to => {
  try { await loadSession(); startup.error = '' }
  catch { startup.error = '无法连接工作台服务，请稍后重试。'; return true }
  if (!session.required && (to.meta.public || to.name === 'account')) return '/'
  if (session.required && !session.user && !to.meta.public) return {name:'login',query:{redirect:to.fullPath}}
  if (session.user && (to.name === 'login' || to.name === 'register')) return '/'
})
window.addEventListener('repromind:auth-expired', () => {
  session.user = null
  if (!router.currentRoute.value.meta.public) void router.replace({name:'login',query:{redirect:router.currentRoute.value.fullPath}})
})
window.addEventListener('pageshow', event => { if (event.persisted) window.location.reload() })

createApp(App).use(router).mount('#app')
