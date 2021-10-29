import { InferenceSession, Tensor } from 'onnxruntime-web'

import modelOnnxUrl from 'src/algorithms/model.onnx'

export interface ModelResult {
  c: number[]
}

export async function runModel(): Promise<ModelResult> {
  const session = await InferenceSession.create(modelOnnxUrl)

  const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
  const a = new Tensor('float32', dataA, [3, 4])
  const b = new Tensor('float32', dataB, [4, 3])

  const feeds = { a, b }

  const results = await session.run(feeds)

  return { c: Array.from(results.c.data as Float32Array) }
}
