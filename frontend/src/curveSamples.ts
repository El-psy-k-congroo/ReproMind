import type { TelemetryPoint } from './types'

export function curveSamples(points: TelemetryPoint[], key: string, min: number, max: number) {
  const epochs = points.map(point => point.epoch).filter(Number.isFinite)
  const first = Math.min(...epochs), last = Math.max(...epochs)
  return points.filter(point => Number.isFinite(point.epoch) && Number.isFinite(point[key])).map(point => ({
    point,
    x: last > first ? 48 + (point.epoch - first) * 388 / (last - first) : 242,
    y: max > min ? 130 - (point[key]! - min) * 100 / (max - min) : 80,
  }))
}

export function nearestSample(samples: { x: number }[], x: number) {
  return samples.reduce((best, point, index) => Math.abs(point.x - x) < Math.abs(samples[best]!.x - x) ? index : best, 0)
}
