import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import DashboardView from './views/DashboardView.vue'
import ProjectView from './views/ProjectView.vue'
import InsightsView from './views/InsightsView.vue'
import './styles.css'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'dashboard', component: DashboardView },
    { path: '/projects/:id', name: 'project', component: ProjectView },
    { path: '/insights', name: 'insights', component: InsightsView },
    { path: '/agents', name: 'agents', component: () => import('./views/AgentsView.vue') },
    { path: '/models', name: 'models', component: () => import('./views/ModelsView.vue') },
  ],
})

createApp(App).use(router).mount('#app')
