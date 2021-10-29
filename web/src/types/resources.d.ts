declare module '*.onnx' {
  const url: string
  export default url
}

declare module '*.json' {
  const content: object
  export default content
}

declare module '*.jpeg' {
  const url: string
  export default url
}

declare module '*.png' {
  const url: string
  export default url
}

declare module '*.gif' {
  const url: string
  export default url
}

declare module '*.webp' {
  const url: string
  export default url
}

declare module '*.svg' {
  import type { PureComponent, HTMLProps, SVGProps } from 'react'

  declare const url: string
  declare class SVG extends PureComponent<SVGProps<SVGElement>> {}
  export { SVG as ReactComponent }
  export default url
}
