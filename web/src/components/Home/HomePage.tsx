import React, { useCallback, useEffect, useState, useRef } from 'react'

import { Col, Container as ContainerBase, Row } from 'reactstrap'
import styled from 'styled-components'

import { ModelResult } from 'src/algorithms/runModel'
import { Uploader } from 'src/components/Common/Uploader'
import { Camera } from 'src/components/Camera/Camera'

const Container = styled(ContainerBase)`
  max-width: ${(props) => props.theme.md};
  flex: 1 0 100%;
`

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

    // RGBA uint8 image, 4 bytes per pixel, color [0..255]
    const imgu8 = ctx.getImageData(0, 0, canvas.width, canvas.height)

    // RGB float32 image, 3 floats per pixel, color values [0..1]
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

    URL.revokeObjectURL(img.src)
  })
  img.src = url
}

export function HomePage() {
  const [result, setResult] = useState<ModelResult>()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const onUpload = useCallback((file: File) => {
    void readImage(file, canvasRef?.current) // eslint-disable-line no-void
  }, [])

  useEffect(() => {
    // eslint-disable-next-line no-void
    void importAndRunModel()
      .then((result) => setResult(result))
      .catch(console.error)
  }, [])

  return (
    <Container>
      <Uploader onUpload={onUpload}>
        <Row noGutters>
          <Col>
            <Camera />
          </Col>
        </Row>

        <Row noGutters>
          <Col>{result && JSON.stringify(result, null, 2)}</Col>
        </Row>

        <Row noGutters>
          <Col>
            <canvas id="canvas" ref={canvasRef} width={200} height={200} style={{ backgroundColor: '#ffff0044' }} />
          </Col>
        </Row>
      </Uploader>
    </Container>
  )
}
