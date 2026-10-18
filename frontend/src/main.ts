import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import DashboardView from './views/DashboardView.vue'
import ProjectView from './views/ProjectView.vue'
import InsightsView from './views/InsightsView.vue'
import './styles.css'
import { loadSession, session, startup } from './session'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', name: 'login', component: () => import('./views/AuthView.vue'), meta: {public:true} },
    { path: '/register', name: 'register', component: () => import('./views/AuthView.vue'), meta: {public:true} },
    { path: '/recover', name: 'recover', component: () => import('./views/AuthView.vue'), meta: {public:true} },
    { path: '/account', name: 'account', component: () => import('./views/AccountView.vue') },
    { path: '/', name: 'dashboard', component: DashboardView },
    { path: '/projects/:id', name: 'project', component: ProjectView },
    { path: '/insights', name: 'insights', component: InsightsView },
    { path: '/agents', name: 'agents', component: () => import('./views/AgentsView.vue') },
    { path: '/models', name: 'models', component: () => import('./views/ModelsView.vue') },
  ],
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
