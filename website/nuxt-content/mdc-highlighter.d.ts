declare module '#mdc-highlighter' {
  type HighlightResult = {
    tree: unknown[]
    className: string
    inlineStyle: string
    style: string
  }

  type Highlighter = (
    code: string,
    lang?: string,
    theme?: unknown,
    options?: {
      highlights?: number[]
      meta?: string
    }
  ) => Promise<HighlightResult>

  const highlighter: Highlighter
  export default highlighter
}
