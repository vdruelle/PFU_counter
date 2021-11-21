import React, { useCallback, useEffect, useState, useRef } from 'react'

import { Button, Col, Container as ContainerBase, Row } from 'reactstrap'
import styled from 'styled-components'

import { ModelResult } from 'src/algorithms/runModel'
import { canvasDrawBox } from 'src/helpers/canvasDrawBox'
import { useViewport } from 'src/helpers/useViewport'
import { usePickFile } from 'src/helpers/usePickFile'
import { Uploader } from 'src/components/Common/Uploader'
import { Camera } from 'src/components/Camera/Camera'
import { DeviceSelector } from 'src/components/Camera/DeviceSelector'
import { useVideoInputDevices } from 'src/components/Camera/useVideoInputDevices'
import { readImageFile } from 'src/helpers/readImageFile'
import { resample } from 'src/helpers/resample'

const Container = styled(ContainerBase)`
  max-width: ${(props) => props.theme.md};
  flex: 1 0 auto;
  display: flex;
  flex-direction: column;
  padding: 0;
`

const CameraWrapper = styled.div`
  flex: 1 0 auto;
  margin-left: 0;
`

const ImageCanvas = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  margin: 0;
  padding: 0;
  z-index: 9;
`

const MainControlsWrapper = styled.div`
  width: 100%;
  height: 100%;
  z-index: 10;
  display: flex;
`

const MainControlsRow = styled.div`
  display: flex;
`

const MainButton = styled(Button)`
  width: 150px;
  height: 80px;
  margin: 5px;
`

export async function importAndRunModel() {
  const { runModel } = await import('src/algorithms/runModel')
  return runModel()
}

export enum Mode {
  Image,
  Camera,
}

export function HomePage() {
  const { devices, currentDevice, onDeviceSelected } = useVideoInputDevices()
  const cameraContainer = useRef<HTMLDivElement>(null)
  const viewport = useViewport(cameraContainer)
  const [result, setResult] = useState<ModelResult>()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { openFileSelector, file, clearFile } = usePickFile('image/*')

  const frame = useRef<ImageData>()
  const [image, setImage] = useState<ImageData | undefined>()
  const [mode, setMode] = useState<Mode>(Mode.Image)
  const toggleCamera = useCallback(() => setMode((mode) => (mode === Mode.Camera ? Mode.Image : Mode.Camera)), [])
  const showCamera = mode === Mode.Camera
  const showCanvas = mode === Mode.Image && image

  const close = useCallback(() => {
    setMode(Mode.Image)
    setImage(undefined)
    clearFile()
  }, [clearFile])

  const onUpload = useCallback((file: File) => {
    setMode(Mode.Image)
    void readImageFile(file).then((image) => setImage(image)) // eslint-disable-line no-void
  }, [])

  useEffect(() => {
    if (file) {
      onUpload(file)
    }
  }, [file, onUpload])

  useEffect(() => {
    if (!image || !canvasRef?.current) {
      return
    }

    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) {
      return
    }

    const { width, height } = canvasRef.current
    const imgu8Resampled = resample(image, width, height)
    ctx.clearRect(0, 0, width, height)
    ctx.putImageData(imgu8Resampled, 0, 0)

    canvasDrawBox({ top: 122, left: 100, width: 400, height: 100 }, '#ff0000', ctx)
    canvasDrawBox({ top: 50, left: 30, width: 100, height: 400 }, '#00ff00', ctx)
  })

  useEffect(() => {
    // eslint-disable-next-line no-void
    void importAndRunModel()
      .then((result) => setResult(result))
      .catch(console.error)
  }, [])

  const onVideoFrame = useCallback((imageData: ImageData) => {
    frame.current = imageData
  }, [])

  const onVideoOverlay = useCallback(
    (overlayCtx: CanvasRenderingContext2D, width: number, height: number) => {
      // const { width: overlayWidth, height: overlayHeight } = overlay.current
      // const { videoWidth, videoHeight, width, height } = camera.current.video
      // const { width: imgWidth, height: imgHeight } = frame

      if (!frame?.current) {
        return
      }

      // const imgu8Resampled = resample(frame.current, 400, 600)

      overlayCtx.clearRect(0, 0, width, height)
      // overlayCtx.putImageData(imgu8Resampled, 0, 0)

      canvasDrawBox({ top: 122, left: 100, width: 400, height: 100 }, '#ff0000', overlayCtx)
      canvasDrawBox({ top: 50, left: 30, width: 100, height: 400 }, '#00ff00', overlayCtx)
    },
    [frame],
  )

  console.info({ result })

  return (
    <Uploader onUpload={onUpload}>
      <Container>
        <Row noGutters>
          <Col>
            {showCamera && (
              <DeviceSelector currentDevice={currentDevice} devices={devices} onDeviceSelected={onDeviceSelected} />
            )}
          </Col>

          <Col className="text-center d-flex">
            <p className="mx-auto my-auto py-1">{'PFU Counter'}</p>
          </Col>

          <Col className="d-flex">
            {(showCamera || showCanvas) && <Button className="flex-1 ml-auto" onClick={close} close />}
          </Col>
        </Row>

        <CameraWrapper ref={cameraContainer}>
          <Row noGutters className="w-100 h-100">
            <Col className="w-100 h-100 d-flex">
              {viewport.width > 0 &&
                viewport.height > 0 &&
                (showCamera ? (
                  <Camera
                    viewport={viewport}
                    currentDevice={currentDevice}
                    onVideoFrame={onVideoFrame}
                    onVideoOverlay={onVideoOverlay}
                  />
                ) : (
                  image && <ImageCanvas ref={canvasRef} width={viewport.width} height={viewport.height} />
                ))}

              {!showCamera && !image && (
                <MainControlsWrapper>
                  <MainControlsRow className="mx-auto my-auto">
                    <MainButton onClick={openFileSelector}>{'Choose image'}</MainButton>
                    <MainButton onClick={toggleCamera}>{'Show camera'}</MainButton>
                  </MainControlsRow>
                </MainControlsWrapper>
              )}
            </Col>
          </Row>
        </CameraWrapper>
      </Container>
    </Uploader>
  )
}
