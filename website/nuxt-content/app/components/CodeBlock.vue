<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import highlighter from '#mdc-highlighter'
import type { MDCParserResult } from '@nuxtjs/mdc'

const props = withDefaults(defineProps<{
  code: string
  language?: string
  filename?: string
  highlights?: number[]
  meta?: string
}>(), {
  language: 'text',
  filename: undefined,
  highlights: () => [],
  meta: ''
})

const parsed = ref<MDCParserResult | null>(null)
const fence = computed(() => props.code.includes('```') ? '````' : '```')
const info_string = computed(() => {
  const parts = [props.language]

  if (props.filename) {
    parts.push(`[${props.filename}]`)
  }

  if (props.meta) {
    parts.push(props.meta)
  }

  return parts.filter(Boolean).join(' ')
})
const markdown = computed(() => {
  return `${fence.value}${info_string.value ? `${info_string.value}\n` : '\n'}${props.code}${props.code.endsWith('\n') ? '' : '\n'}${fence.value}`
})

async function update_highlighted_code(): Promise<void> {
  parsed.value = await parseMarkdown(markdown.value, {
    highlight: {
      highlighter: highlighter as never
    }
  })
}

await update_highlighted_code()

watch(
  () => [markdown.value, props.highlights.join(',')],
  () => {
    void update_highlighted_code()
  }
)
</script>

<template>
  <MDCRenderer
    v-if="parsed"
    :body="parsed.body"
    :data="parsed.data"
  />
</template>
