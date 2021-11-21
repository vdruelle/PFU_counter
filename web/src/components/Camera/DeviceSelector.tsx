import React, { useCallback, useState } from 'react'
import { Dropdown, DropdownItem, DropdownMenu, DropdownToggle } from 'reactstrap'

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
      <DropdownToggle caret color="transparent">
        {currentDevice?.label ?? ''}
      </DropdownToggle>
      <DropdownMenu>
        {devices.map(({ deviceId, label }) => (
          <DropdownItem key={deviceId}>{label ?? deviceId}</DropdownItem>
        ))}
      </DropdownMenu>
    </Dropdown>
  )
}
