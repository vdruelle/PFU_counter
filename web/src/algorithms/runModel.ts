import { InferenceSession, Tensor } from 'onnxruntime-web'

import modelOnnxUrl from 'src/algorithms/model.onnx'

export interface ModelResult {
  c: number[]
}

export class Model {
  session: InferenceSession | undefined
  once = false

  public async init() {
    if (!this.once) {
      this.session = await InferenceSession.create(modelOnnxUrl)
      this.once = true
    }
    return this
  }

  public teardown() {
    delete this.session
  }

  public async run(imageData: ImageData): Promise<ModelResult> {
    if (!this.session) {
      throw new Error('Internal error: Model session is not ready.')
    }

    const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
    const a = new Tensor('float32', dataA, [3, 4])
    const b = new Tensor('float32', dataB, [4, 3])

    const feeds = { a, b }
    const results = await this.session.run(feeds)
    const c = Array.from(results.c.data as Float32Array)

    return { c }
  }
}
