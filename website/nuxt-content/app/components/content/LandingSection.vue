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

type Feature = {
  icon: string
  title: string
  description: string
}

defineProps<{
  title: string
  links?: ButtonLink[]
  features?: Feature[]
}>()
</script>

<template>
  <UPageSection
    class="dark:bg-neutral-950"
    :title="title"
  >
    <template #links>
      <UButton
        v-for="(link, index) in links"
        :key="index"
        v-bind="link"
      />
    </template>

    <template #features>
      <UPageFeature
        v-for="(feature, index) in features"
        :key="index"
        :icon="feature.icon"
        :title="feature.title"
        :description="feature.description"
      />
    </template>
  </UPageSection>
</template>
