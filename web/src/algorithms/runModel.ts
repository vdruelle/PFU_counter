import { InferenceSession, Tensor } from 'onnxruntime-web'

import { resample } from 'src/helpers/resample'
import { u8tof32 } from 'src/helpers/u8tof32'

import colonyCounterOnnxUrl from '../../../model_saves/Colony_counter.onnx'
import plateDetectorOnnxUrl from '../../../model_saves/Plate_detector.onnx'

abstract class OnnxModel {
  session: InferenceSession | undefined
  once = false
  modelOnnxUrl: string

  constructor(modelOnnxUrl: string) {
    this.modelOnnxUrl = modelOnnxUrl
  }

  public async init() {
    if (!this.once) {
      this.session = await InferenceSession.create(this.modelOnnxUrl)
      this.once = true
    }
    return this
  }

  public teardown() {
    delete this.session
  }
}

// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface ColonyCounterResult {}

class ColonyCounter extends OnnxModel {
  constructor() {
    super(colonyCounterOnnxUrl)
  }

  public async run(imageData: ImageData): Promise<ColonyCounterResult> {
    if (!this.session) {
      throw new Error('Internal error: Model session is not ready.')
    }

    const width = 400
    const height = 400

    const imageDataF32 = u8tof32(resample(imageData, height, width))
    const imageTensor = new Tensor('float32', imageDataF32.data, [3, height, width])

    const feeds = { Spot_image: imageTensor }
    return this.session.run(feeds)
  }
}

// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface PlateDetectorResult {}

class PlateDetector extends OnnxModel {
  constructor() {
    super(plateDetectorOnnxUrl)
  }

  public async run(imageData: ImageData): Promise<PlateDetectorResult> {
    if (!this.session) {
      throw new Error('Internal error: Model session is not ready.')
    }

    const width = 3456
    const height = 4608

    const imageDataF32 = u8tof32(resample(imageData, height, width))
    const imageTensor = new Tensor('float32', imageDataF32.data, [3, height, width])

    const feeds = { Plate_image: imageTensor }
    return this.session.run(feeds)
  }
}

export interface ModelResult {
  colonyCounterResult: ColonyCounterResult
  plateDetectorResult: PlateDetectorResult
}

export class Model {
  colonyCounter: ColonyCounter
  plateDetector: PlateDetector

  constructor() {
    this.colonyCounter = new ColonyCounter()
    this.plateDetector = new PlateDetector()
  }

  public async init() {
    await this.colonyCounter.init()
    await this.plateDetector.init()
    return this
  }

  public teardown() {
    this.colonyCounter.teardown()
    this.plateDetector.teardown()
  }

  public async run(imageData: ImageData): Promise<ModelResult> {
    // const colonyCounterResult = await this.colonyCounter.run(imageData)
    // const plateDetectorResult = await this.plateDetector.run(imageData)

    const [colonyCounterResult, plateDetectorResult] = await Promise.all([
      this.colonyCounter.run(imageData),
      this.plateDetector.run(imageData),
    ])

    return { colonyCounterResult, plateDetectorResult }
  }
}
