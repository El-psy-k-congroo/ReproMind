export function installationTransportFailed(error: string, failure: unknown): boolean {
  const kind = failure && typeof failure === 'object' && 'kind' in failure ? failure.kind : ''
  return kind === 'network' || /ProxyError|Cannot connect to proxy|ReadTimeout|ConnectionError|Connection reset|CERTIFICATE_VERIFY_FAILED|SSLError/i.test(error)
}

export function installationIndexUnavailable(error: string, failure: unknown, logs = ''): boolean {
  const kind = failure && typeof failure === 'object' && 'kind' in failure ? failure.kind : ''
  if (kind === 'index_unavailable') return true
  if (kind && kind !== 'command_error') return false
  return /No repodata found|repodata[^\n]*(?:not found|missing|unavailable)/i.test(`${error}\n${logs}`)
}

export function tensorflowFailureHeading(error: string, failure: unknown, logs = ''): string {
  if (installationIndexUnavailable(error, failure, logs)) return '上次 TensorFlow GPU 环境依赖预检未完成：软件源索引未加载。'
  if (installationTransportFailed(error, failure)) return 'TensorFlow GPU 环境准备未完成：软件源连接失败。'
  if (/返回码 124|命令超过 \d+ 秒|安装命令超过 \d+ 秒/.test(error)) {
    const seconds = error.match(/(?:安装)?命令超过 (\d+) 秒/)?.[1]
    return `TensorFlow GPU 环境安装超时${seconds ? `（${Math.round(Number(seconds) / 60)} 分钟）` : ''}，已停止安装，未启动训练。`
  }
  return 'TensorFlow GPU 环境准备未完成，请查看诊断并重新核验依赖。'
}
