declare module '*.md' {
  const MDXComponent: () => JSX.Element
  export default MDXComponent
}

declare module '*.mdx' {
  const MDXComponent: () => JSX.Element
  export default MDXComponent
}
