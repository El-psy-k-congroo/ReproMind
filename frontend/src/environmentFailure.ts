export function installationTransportFailed(error: string, failure: unknown): boolean {
  const kind = failure && typeof failure === 'object' && 'kind' in failure ? failure.kind : ''
  return kind === 'network' || /ProxyError|Cannot connect to proxy|ReadTimeout|ConnectionError|Connection reset|CERTIFICATE_VERIFY_FAILED|SSLError/i.test(error)
}
