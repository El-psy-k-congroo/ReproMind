// A running tab can still reference chunks removed by a newer deployment.
// Retry once per destination and minute, using tab-local storage to stop reload loops.
export function recoverRouteImport(
  error: unknown,
  target: string,
  storage: Pick<Storage, 'getItem' | 'setItem'>,
  navigate: (target: string) => void,
  now = Date.now(),
): 'unrelated' | 'reloading' | 'failed' {
  const message = error instanceof Error ? error.message : String(error)
  if (!/Failed to fetch dynamically imported module|Importing a module script failed|error loading dynamically imported module|Unable to preload CSS/i.test(message)) return 'unrelated'
  const key = `repromind:route-recovery:${target}`
  try {
    const previous = Number(storage.getItem(key))
    if (previous && now - previous < 60_000) return 'failed'
    storage.setItem(key, String(now))
    navigate(target)
    return 'reloading'
  } catch { return 'failed' }
}
