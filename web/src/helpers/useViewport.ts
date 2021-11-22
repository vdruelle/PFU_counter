import { useRef } from 'react'
import { useResizeDetector } from 'react-resize-detector'

export interface UseViewportResult {
  width: number
  height: number
}

export function useViewport(targetRef: ReturnType<typeof useRef>): UseViewportResult {
  const resize = useResizeDetector({ targetRef, handleHeight: true, handleWidth: true })
  return {
    width: resize.width ?? 0,
    height: resize.height ?? 0,
  }
}
