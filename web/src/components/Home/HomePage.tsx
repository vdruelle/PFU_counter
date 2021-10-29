import React, { useCallback, useEffect, useState, useRef } from 'react'

import { Col, Container, Row } from 'reactstrap'

import { ModelResult } from 'src/algorithms/runModel'
import { Uploader } from 'src/components/Common/Uploader'

export async function importAndRunModel() {
  const { runModel } = await import('src/algorithms/runModel')
  return runModel()
}

export async function readImage(file: File, canvas: HTMLCanvasElement | null) {
  if (!canvas) {
    throw new Error('Canvas element is not ready')
  }

  const ctx = canvas.getContext('2d')
  if (!ctx) {
    throw new Error('Cannot get canvas context')
  }

  ctx.fillStyle = 'black'
  const url = URL.createObjectURL(file)
  const img = new Image()
  img.addEventListener('load', () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    canvas.width = img.width
    canvas.height = img.height
    ctx.drawImage(img, 0, 0, img.width, img.height)
    URL.revokeObjectURL(img.src)
  })
  img.src = url
}

export function HomePage() {
  const [result, setResult] = useState<ModelResult>()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const onUpload = useCallback((file: File) => {
    void readImage(file, canvasRef?.current)
  }, [])

  useEffect(() => {
    // eslint-disable-next-line no-void
    void importAndRunModel()
      .then((result) => setResult(result))
      .catch(console.error)
  }, [])

  return (
    <Container>
      <Row noGutters>
        <Col>
          <Uploader onUpload={onUpload} />
        </Col>
      </Row>

      <Row noGutters>
        <Col>{result && JSON.stringify(result, null, 2)}</Col>
      </Row>

      <Row noGutters>
        <Col>
          <canvas id="canvas" ref={canvasRef} width={200} height={200} />
        </Col>
      </Row>
    </Container>
  )
}
