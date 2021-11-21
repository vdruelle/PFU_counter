import React from 'react'
import { Button, ButtonProps } from 'reactstrap'

import { MdClose } from 'react-icons/md'

export function CloseButton(props: ButtonProps) {
  return (
    <Button {...props}>
      <MdClose size={24} />
    </Button>
  )
}
