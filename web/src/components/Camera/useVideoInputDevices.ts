/* eslint-disable no-void,promise/always-return */
import React, { useCallback } from 'react'

export function useVideoInputDevices() {
  const [devices, setDevices] = React.useState<MediaDeviceInfo[]>([])
  const [currentDevice, setCurrentDevice] = React.useState<MediaDeviceInfo | undefined>(undefined)

  React.useEffect(() => {
    void navigator.mediaDevices.enumerateDevices().then((mediaDevices: MediaDeviceInfo[]) => {
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

  return { devices, currentDevice, onDeviceSelected }
}
