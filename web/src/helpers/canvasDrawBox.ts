export interface Box {
  top: number
  left: number
  width: number
  height: number
}

export function canvasDrawBox(boundingBox: Box, color: string, ctx: CanvasRenderingContext2D) {
  ctx.beginPath()
  ctx.rect(boundingBox.left, boundingBox.top, boundingBox.width, boundingBox.height)
  ctx.strokeStyle = color
  ctx.stroke()
}
