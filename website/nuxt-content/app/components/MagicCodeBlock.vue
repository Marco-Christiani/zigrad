<script setup lang="ts">
import { ref } from 'vue'
import { ShikiMagicMove } from 'shiki-magic-move/vue'
import type { HighlighterCore } from 'shiki/core'

import { get_magic_move_highlighter } from '../utils/magic-move-highlighter'

const props = withDefaults(defineProps<{
  code: string
  language?: string
  filename?: string
}>(), {
  language: 'text',
  filename: undefined
})

const highlighter = ref<HighlighterCore | null>(null)

onMounted(async () => {
  highlighter.value = await get_magic_move_highlighter()
})
</script>

<template>
  <ClientOnly>
    <CodeBlock
      v-if="!highlighter"
      :code="code"
      :language="language"
      :filename="filename"
    />

    <div
      v-else
      class="my-5"
    >
      <div class="relative group">
        <div
          v-if="filename"
          class="rounded-t-md border border-muted border-b-0 bg-default px-4 py-3"
        >
          <span class="text-sm/6 text-default">{{ filename }}</span>
        </div>

        <div
          class="overflow-x-auto border border-muted bg-muted px-4 py-3"
          :class="filename ? 'rounded-t-none rounded-b-md' : 'rounded-md'"
        >
          <ShikiMagicMove
            :highlighter="highlighter"
            :code="code"
            :lang="language"
            theme="material-theme-palenight"
            :options="{
              duration: 700,
              stagger: 0.15,
              animateContainer: true
            }"
          />
        </div>
      </div>
    </div>

    <template #fallback>
      <CodeBlock
        :code="code"
        :language="language"
        :filename="filename"
      />
    </template>
  </ClientOnly>
</template>
