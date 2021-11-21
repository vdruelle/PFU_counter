import React, { PropsWithChildren, useState } from 'react'

import styled, { DefaultTheme } from 'styled-components'
import { FileRejection, useDropzone } from 'react-dropzone'

import { theme } from 'src/theme'

export type UpdateErrorsFunction = (prevErrors: string[]) => string[]

export interface MakeOnDropParams {
  onUpload(file: File): void

  setErrors(updateErrors: string[] | UpdateErrorsFunction): void
}

export function makeOnDrop({ onUpload, setErrors }: MakeOnDropParams) {
  function handleError(error: Error) {
    if (error instanceof UploadErrorTooManyFiles) {
      setErrors((prevErrors) => [...prevErrors, 'Only one file is expected'])
    } else if (error instanceof UploadErrorUnknown) {
      setErrors((prevErrors) => [...prevErrors, 'Unknown error'])
    } else {
      throw error
    }
  }

  async function processFiles(acceptedFiles: File[], rejectedFiles: FileRejection[]) {
    const nFiles = acceptedFiles.length + rejectedFiles.length

    if (nFiles > 1) {
      throw new UploadErrorTooManyFiles(nFiles)
    }

    if (acceptedFiles.length !== 1) {
      throw new UploadErrorTooManyFiles(acceptedFiles.length)
    }

    const file = acceptedFiles[0]
    onUpload(file)
  }

  return async function onDrop(acceptedFiles: File[], rejectedFiles: FileRejection[]) {
    setErrors([])
    try {
      await processFiles(acceptedFiles, rejectedFiles)
    } catch (error) {
      handleError(error)
    }
  }
}

export enum UploadZoneState {
  normal = 'normal',
  accept = 'accept',
  reject = 'reject',
  hover = 'hover',
}

class UploadErrorTooManyFiles extends Error {
  public readonly nFiles: number

  constructor(nFiles: number) {
    super(`when uploading: one file is expected, but got ${nFiles}`)
    this.nFiles = nFiles
  }
}

class UploadErrorUnknown extends Error {
  constructor() {
    super(`when uploading: unknown error`)
  }
}

export type UploadZoneElems = keyof typeof theme.uploadZone

export function getUploadZoneTheme(props: { state: UploadZoneState; theme: DefaultTheme }, elem: UploadZoneElems) {
  return props.theme.uploadZone[elem][props.state]
}

export const UploadZoneWrapper = styled.div`
  width: 100%;
  height: 100%;

  &:focus-within {
    border: none;
    inset: none;
    border-image: none;
  }
`

export const UploadZone = styled.div<{ state: UploadZoneState }>`
  display: flex;
  height: 100%;
  border-radius: 5px;
  border: ${(props) => getUploadZoneTheme(props, 'border')};
  color: ${(props) => getUploadZoneTheme(props, 'color')};
  background-color: ${(props) => getUploadZoneTheme(props, 'background')};
  box-shadow: ${(props) => getUploadZoneTheme(props, 'box-shadow')};
`

export interface UploaderGenericProps {
  onUpload(file: File): void
}

export function Uploader({ onUpload, children }: PropsWithChildren<UploaderGenericProps>) {
  const [errors, setErrors] = useState<string[]>([])

  const { getRootProps, isDragAccept, isDragReject } = useDropzone({
    onDrop: makeOnDrop({ onUpload, setErrors }),
    multiple: false,
  })

  const hasErrors = errors.length > 0

  if (hasErrors) {
    console.warn(`Errors when uploading:\n${errors.join('\n')}`)
  }

  let state = UploadZoneState.normal
  if (isDragAccept) state = UploadZoneState.accept
  else if (isDragReject) state = UploadZoneState.reject

  return (
    <UploadZoneWrapper {...getRootProps()}>
      <UploadZone state={state}>{children}</UploadZone>
    </UploadZoneWrapper>
  )
}
