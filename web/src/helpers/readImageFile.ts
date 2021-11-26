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

      URL.revokeObjectURL(img.src)
    })
    img.src = url
  })
}
