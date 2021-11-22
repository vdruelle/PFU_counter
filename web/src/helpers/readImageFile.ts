export async function readImageFile(file: File): Promise<ImageData> {
  return new Promise<ImageData>((resolve) => {
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.addEventListener('load', () => {
      const canvas = document.createElement('canvas')
      if (!canvas) {
        throw new Error('Canvas element is not ready')
      }

      const ctx = canvas.getContext('2d')
      if (!ctx) {
        throw new Error('Cannot get canvas context')
      }

      canvas.width = img.width
      canvas.height = img.height

      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0, img.width, img.height)

      const imgu8 = ctx.getImageData(0, 0, canvas.width, canvas.height)
      resolve(imgu8)

      // // RGB float32 image, 3 floats per pixel, color values [0..1]
      // const imgf32 = new Float32Array(imgu8.height * imgu8.width * 3)
      //
      // for (let y = 0; y < imgu8.height; y += 1) {
      //   for (let x = 0; x < imgu8.width; x += 1) {
      //     const r = imgu8.data[x * imgu8.height + y * 4 + 0]
      //     const g = imgu8.data[x * imgu8.height + y * 4 + 1]
      //     const b = imgu8.data[x * imgu8.height + y * 4 + 2]
      //
      //     imgf32[x * imgu8.height + y * 3 + 0] = r / 255
      //     imgf32[x * imgu8.height + y * 3 + 1] = g / 255
      //     imgf32[x * imgu8.height + y * 3 + 2] = b / 255
      //   }
      // }

      URL.revokeObjectURL(img.src)
    })
    img.src = url
  })
}
