import React, { useCallback, useEffect, useMemo, useRef } from 'react'

import styled from 'styled-components'
import WebcamBase from 'react-webcam'

import { UseViewportResult } from 'src/helpers/useViewport'

const Webcam = styled(WebcamBase)`
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  background-color: #000000;
  margin: 0;
  padding: 0;
`

const WebcamOverlay = styled.canvas`
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  margin: 0;
  padding: 0;
`

export interface CameraProps {
  viewport: UseViewportResult
  currentDevice?: MediaDeviceInfo
  onVideoFrame(imageData: ImageData): void
  onVideoOverlay(overlayCtx: CanvasRenderingContext2D, width: number, height: number): void
}

export function Camera({ viewport, currentDevice, onVideoFrame, onVideoOverlay }: CameraProps) {
  const camera = useRef<WebcamBase>(null)
  const overlay = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>()

  const animate = useCallback(() => {
    rafRef.current = requestAnimationFrame(animate)

    if (camera && camera.current && camera.current?.video?.readyState === 4) {
      const cameraCanvas = camera?.current?.getCanvas()
      if (!cameraCanvas) {
        return
      }

      const cameraCtx = cameraCanvas.getContext('2d')
      if (!cameraCtx) {
        return
      }

      const { width, height } = camera.current.video
      const imgu8 = cameraCtx.getImageData(0, 0, width, height)
      onVideoFrame(imgu8)

      if (!overlay || !overlay.current) {
        return
      }

      const overlayCtx = overlay.current.getContext('2d')
      if (!overlayCtx) {
        return
      }

      onVideoOverlay(overlayCtx, overlay.current.width, overlay.current.height)
    }
  }, [onVideoFrame, onVideoOverlay])

  useEffect(() => {
    rafRef.current = requestAnimationFrame(animate)
    return () => {
      if (rafRef && rafRef.current) {
        cancelAnimationFrame(rafRef?.current)
      }
    }
  }, [animate])

  const videoConstraints: MediaTrackConstraints = useMemo(
    () => ({ deviceId: currentDevice?.deviceId }),
    [currentDevice],
  )

  if (!currentDevice) {
    return null
  }

  animate()

  return (
    <>
      <Webcam
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        ref={camera}
        width={viewport.width}
        height={viewport.height}
        audio={false}
        mirrored={false}
        imageSmoothing={false}
        videoConstraints={videoConstraints}
      />
      <WebcamOverlay ref={overlay} width={viewport.width} height={viewport.height} />
    </>
  )
}
