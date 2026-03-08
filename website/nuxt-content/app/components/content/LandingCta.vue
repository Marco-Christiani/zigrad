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
  <UPageSection class="dark:bg-gradient-to-b from-neutral-950 to-neutral-900">
    <UPageCTA
      :title="title"
      :description="description"
      :links="links"
      class="dark:bg-neutral-950"
    >
      <StarsBg />
    </UPageCTA>
  </UPageSection>
</template>
