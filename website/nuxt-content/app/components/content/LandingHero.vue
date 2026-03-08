<script setup lang="ts">
type ButtonLink = {
  label: string
  to: string
  target?: string
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl'
  color?: 'primary' | 'secondary' | 'success' | 'info' | 'warning' | 'error' | 'neutral'
  variant?: 'solid' | 'outline' | 'soft' | 'subtle' | 'ghost' | 'link'
  icon?: string
  trailingIcon?: string
}

defineProps<{
  title: string
  description: string
  links?: ButtonLink[]
}>()
</script>

<template>
  <UPageHero
    orientation="horizontal"
    class="dark:bg-gradient-to-b from-neutral-900 to-neutral-950"
  >
    <template #top>
      <HeroBackground />
    </template>

    <template #title>
      {{ title }}
    </template>

    <template #description>
      {{ description }}
    </template>

    <template #links>
      <UButton
        v-for="(link, index) in links"
        :key="index"
        v-bind="link"
      />
    </template>

    <slot />
  </UPageHero>
</template>
