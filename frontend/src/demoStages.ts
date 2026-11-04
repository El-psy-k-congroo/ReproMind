export const demoStages = [
  { id: 'evidence', title: '论文与实验目标', seconds: 40 },
  { id: 'preparation', title: '源码、环境与数据', seconds: 60 },
  { id: 'execution', title: '执行命令与训练日志', seconds: 50 },
  { id: 'coding', title: '代码修改与校验', seconds: 70 },
  { id: 'results', title: '结果、曲线与报告', seconds: 80 },
] as const

export function resolveRunId(history: Array<{ id: string }>, selected: string, requested: string): string {
  if (requested) {
    if (!history.some(item => item.id === requested)) throw new Error('链接指定的实验记录不存在，请返回项目选择记录。')
    return requested
  }
  return history.some(item => item.id === selected) ? selected : history[0]?.id || ''
}

export function demoLink(projectId: string, runId: string, stage = ''): string {
  const query = new URLSearchParams({ run: runId })
  if (stage) query.set('stage', stage)
  return `/projects/${encodeURIComponent(projectId)}/demo?${query}`
}
