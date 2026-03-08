import { createHighlighter } from 'shiki'
import type { HighlighterCore } from 'shiki/core'

import mlir_grammar from '../../shiki-grammars/mlir-grammar.json'
import pdll_grammar from '../../shiki-grammars/pdll-grammar.json'
import tablegen_grammar from '../../shiki-grammars/tablegen-grammar.json'
import zxpr_grammar from '../../shiki-grammars/zxpr-grammar.json'

let highlighter_promise: Promise<HighlighterCore> | null = null

async function create_magic_move_highlighter(): Promise<HighlighterCore> {
  const highlighter = await createHighlighter({
    themes: [
      'material-theme-lighter',
      'material-theme',
      'material-theme-palenight'
    ],
    langs: [
      'js',
      'jsx',
      'json',
      'ts',
      'tsx',
      'vue',
      'css',
      'html',
      'bash',
      'md',
      'mdc',
      'yaml',
      'zig',
      'llvm'
    ]
  })

  await highlighter.loadLanguage({
    ...mlir_grammar,
    aliases: ['mlir']
  } as never)

  await highlighter.loadLanguage({
    ...pdll_grammar,
    aliases: ['pdll']
  } as never)

  await highlighter.loadLanguage({
    ...tablegen_grammar,
    aliases: ['tablegen', 'td']
  } as never)

  await highlighter.loadLanguage({
    ...zxpr_grammar,
    aliases: ['zxpr']
  } as never)

  return highlighter
}

export function get_magic_move_highlighter(): Promise<HighlighterCore> {
  highlighter_promise ??= create_magic_move_highlighter()
  return highlighter_promise
}
