/** Device initialization and CUDA enumeration are not training proof. */
export function trainingDeviceLabel(rows: Array<{ kind: string }> = []): string {
  if (rows.some(row => row.kind === 'gpu_compute_placement')) return '本次实验日志已记录 GPU 计算算子'
  if (rows.some(row => row.kind === 'training_device_gpu')) return '本次实验日志已记录 GPU 训练设备'
  if (rows.some(row => row.kind === 'training_device_cpu')) return '本次实验日志已记录 CPU 训练设备'
  return '本次实验尚无明确训练设备证据'
}
