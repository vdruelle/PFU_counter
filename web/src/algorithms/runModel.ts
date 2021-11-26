import { InferenceSession, Tensor } from 'onnxruntime-web'

// import modelOnnxUrl from 'src/algorithms/Colony_counter.onnx'
import modelOnnxUrl from 'src/algorithms/Plate_detector.onnx'
import { resample } from 'src/helpers/resample'
import { u8tof32 } from 'src/helpers/u8tof32'

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

    // const width = 400
    // const height = 400

    const width = 3456
    const height = 4608

    const imageDataF32 = u8tof32(resample(imageData, height, width))
    const imageTensor = new Tensor('float32', imageDataF32.data, [3, height, width])

    // const feeds = { Spot_image: imageTensor }
    const feeds = { Plate_image: imageTensor }

    const results = await this.session.run(feeds)

    console.log({ results })
    console.log(JSON.stringify(results, null, 2))
    // const c = Array.from(results.c.data as Float32Array)

    return { c: [] }
  }
}
