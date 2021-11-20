/* eslint-disable promise/always-return,@typescript-eslint/no-floating-promises,promise/catch-or-return */
import React, { useCallback, useMemo, useState } from 'react'

import Webcam from 'react-webcam'
import { Col, Row, Dropdown, DropdownItem, DropdownMenu, DropdownToggle } from 'reactstrap'
import { resample } from 'src/helpers/resample'

export interface BoundingBox {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

export function drawBoundingBox(boundingBox: BoundingBox, color: string, ctx: CanvasRenderingContext2D) {
  ctx.rect(boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX, boundingBox.maxY - boundingBox.minY)
  ctx.strokeStyle = color
  ctx.stroke()
}

export interface DeviceSelectorProps {
  currentDevice?: MediaDeviceInfo
  devices: MediaDeviceInfo[]
  onDeviceSelected(device: MediaDeviceInfo): void
}

export function DeviceSelector({ currentDevice, devices, onDeviceSelected }: DeviceSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const toggle = useCallback(() => {
    setIsOpen((isOpen) => !isOpen)
  }, [])

  return (
    <Dropdown isOpen={isOpen} toggle={toggle}>
      <DropdownToggle caret>{currentDevice?.label ?? ''}</DropdownToggle>
      <DropdownMenu>
        {devices.map(({ deviceId, label }) => (
          <DropdownItem key={deviceId}>{label ?? deviceId}</DropdownItem>
        ))}
      </DropdownMenu>
    </Dropdown>
  )
}

export function Camera() {
  const camera = React.useRef<Webcam>(null)
  const overlay = React.useRef<HTMLCanvasElement>(null)
  const [currentDevice, setCurrentDevice] = React.useState<MediaDeviceInfo | undefined>(undefined)
  const [devices, setDevices] = React.useState<MediaDeviceInfo[]>([])

  React.useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((mediaDevices: MediaDeviceInfo[]) => {
      const devices = mediaDevices.filter(({ kind }) => kind === 'videoinput')
      setDevices(devices)
      if (devices.length > 0 && devices[0]) {
        setCurrentDevice(devices[0])
      }
    })
  }, [setDevices, setCurrentDevice])

  const onDeviceSelected = useCallback(
    (device: MediaDeviceInfo) => {
      setCurrentDevice(device)
    },
    [setCurrentDevice],
  )

  const videoConstraints: MediaTrackConstraints = useMemo(
    () => ({ deviceId: currentDevice?.deviceId }),
    [currentDevice],
  )

  const rafRef = React.useRef<number>()

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
      let imgu8 = cameraCtx.getImageData(0, 0, width, height)

      if (!overlay || !overlay.current) {
        return
      }

      const overlayCtx = overlay.current.getContext('2d')
      if (!overlayCtx) {
        return
      }

      const { width: overlayWidth, height: overlayHeight } = overlay.current
      // const { videoWidth, videoHeight, width, height } = camera.current.video
      // const { width: imgWidth, height: imgHeight } = imgu8

      imgu8 = resample(imgu8, overlayWidth, overlayHeight)
      overlayCtx.putImageData(imgu8, 0, 0)

      drawBoundingBox({ minX: 0, maxX: 100, minY: 70, maxY: 100 }, '#aa00aa', overlayCtx)
    }
  }, [])

  React.useEffect(() => {
    rafRef.current = requestAnimationFrame(animate)
    return () => {
      if (rafRef && rafRef.current) {
        cancelAnimationFrame(rafRef?.current)
      }
    }
  }, [animate])

  animate()

  return (
    <Row noGutters>
      <Col>
        <Row noGutters>
          <Col>
            <DeviceSelector currentDevice={currentDevice} devices={devices} onDeviceSelected={onDeviceSelected} />
          </Col>
        </Row>

        <Row noGutters>
          <Col>
            {currentDevice && (
              <Webcam
                ref={camera}
                style={{
                  position: 'absolute',
                  top: 0,
                }}
                width={420}
                height={320}
                audio={false}
                mirrored={false}
                imageSmoothing={false}
                videoConstraints={videoConstraints}
              />
            )}
            <div
              style={{ position: 'absolute', top: 0, left: 0, width: 320, height: 320, backgroundColor: '#ff000055' }}
            />
          </Col>

          <Col>
            <canvas
              ref={overlay}
              style={{
                position: 'relative',
                top: 0,
                background: '#0000ff99',
                width: 200,
                height: 200,
              }}
              width={200}
              height={200}
            />
          </Col>
        </Row>
      </Col>
    </Row>
  )
}
