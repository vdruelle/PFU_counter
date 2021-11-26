export interface ImageDataF32 {
  data: Float32Array
  width: number
  height: number
}

// RGB float32 image, 3 floats per pixel, color values [0..1]
export function u8tof32(imgu8: ImageData): ImageDataF32 {
  const imgf32 = new Float32Array(imgu8.height * imgu8.width * 3)

  for (let y = 0; y < imgu8.height; y += 1) {
    for (let x = 0; x < imgu8.width; x += 1) {
      const r = imgu8.data[x * imgu8.height + y * 4 + 0]
      const g = imgu8.data[x * imgu8.height + y * 4 + 1]
      const b = imgu8.data[x * imgu8.height + y * 4 + 2]

      imgf32[x * imgu8.height + y * 3 + 0] = r / 255
      imgf32[x * imgu8.height + y * 3 + 1] = g / 255
      imgf32[x * imgu8.height + y * 3 + 2] = b / 255
    }
  }

  return {
    data: imgf32,
    width: imgu8.width,
    height: imgu8.height,
  }
}
