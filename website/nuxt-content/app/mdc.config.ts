import { defineConfig } from '@nuxtjs/mdc/config'

import mlir_grammar from '../shiki-grammars/mlir-grammar.json'
import pdll_grammar from '../shiki-grammars/pdll-grammar.json'
import tablegen_grammar from '../shiki-grammars/tablegen-grammar.json'
import zxpr_grammar from '../shiki-grammars/zxpr-grammar.json'

export default defineConfig({
  shiki: {
    async setup(highlighter) {
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
    }
  }
})
