/* eslint-disable no-bitwise,unicorn/prefer-math-trunc */
export function round(val: number) {
  return (val + 0.49) << 0
}

export function resample(src: ImageData, width: number, height: number) {
  const dst = new ImageData(width, height)
  const scaleX = src.width / dst.width
  const scaleY = src.height / dst.height
  let pos = 0
  for (let y = 0; y < dst.height; y++) {
    for (let x = 0; x < dst.width; x++) {
      const srcX = round(x * scaleX)
      const srcY = round(y * scaleY)
      const srcPos = (srcY * src.width + srcX) * 4
      dst.data[pos + 0] = src.data[srcPos + 0]
      dst.data[pos + 1] = src.data[srcPos + 1]
      dst.data[pos + 2] = src.data[srcPos + 2]
      dst.data[pos + 3] = src.data[srcPos + 3]
      pos += 4
    }
  }
  return dst
}
