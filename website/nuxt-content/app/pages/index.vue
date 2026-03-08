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

type Section = {
  headline: string
  title: string
  description: string
  link: ButtonLink
  features: Feature[]
}

const title = 'Zigrad'
const description = 'Zigrad is a programmable deep learning framework and compiler that bridges research workflows and production execution.'

const hero_links: ButtonLink[] = [
  { label: 'Explore the pipeline', to: '/pipeline', size: 'xl', trailingIcon: 'i-lucide-arrow-right' },
  { label: 'View on GitHub', to: 'https://github.com/Marco-Christiani/Zigrad', target: '_blank', color: 'neutral', variant: 'outline', size: 'xl', icon: 'i-simple-icons-github' }
]

const sections: Section[] = [
  {
    headline: 'Programmable program representation',
    title: 'One program representation, multiple compilation paths.',
    description:
      'Models are represented in a structured program representation that can be inspected, transformed, and lowered to different compiler backends without rewriting model code.',
    link: { label: 'Explore Architecture', to: '/getting-started', color: 'neutral', variant: 'subtle', size: 'lg', trailingIcon: 'i-lucide-arrow-right' },
    features: [
      // {
      //   icon: 'i-lucide-spline',
      //   title: 'Inspectable program structure',
      //   description: 'Inspect tensor operations and graph structure throughout compilation.'
      // },
      {
       // icon: 'i-lucide-git-branch-plus',
        icon: 'i-lucide-spline',
        title: 'Programmable graph transformations',
        description: 'Apply custom passes to rewrite and optimize the program representation before backend lowering.'
      },
      {
        icon: 'i-lucide-activity',
        title: 'Differentiable program representation',
        description: 'Automatic differentiation is abstracted above the lowering and compilation path.'
      },
      {
        icon: 'i-lucide-layers',
        title: 'Backend-agnostic semantics',
        description: 'The program representation encodes domain-level operations without assuming a compilation target, enabling retargeting without rewriting model code.'
      }
    ]
  },
  {
    headline: 'Retargeting and selective kernelization',
    title: 'Retarget the same program and specialize only the regions that matter.',
    description:
      'The same program can target different compiler backends while selectively replacing regions with specialized kernels when profitability warrants it.',
    link: { label: 'Open explorer', to: '/pipeline', color: 'neutral', variant: 'subtle', size: 'lg', trailingIcon: 'i-lucide-arrow-right' },
    features: [
      {
        icon: 'i-lucide-cpu',
        title: 'Backend retargeting',
        description: 'Execute the same program across compiler backends such as XLA for large-scale training or IREE edge deployment.'
      },
      // {
      //   icon: 'i-lucide-box',
      //   title: 'Match and rewrite',
      //   description: 'Extract regions of the program on the hot path and replace them with specialized kernels.'
      // },
      {
        // icon: 'i-lucide-scroll-text',
        icon: 'i-lucide-scan-search',
        title: 'External kernel providers',
        description: 'Generate kernels with autotuning systems such as TVM or the Mirage Superoptimizer.'
      },
      {
        icon: 'i-lucide-scroll-text',
        title: 'Profitability-gated specialization',
        description: 'Identify candidate regions via pattern matching or compiler hints and rewrite them with specialized kernels only when profiling justifies it.'
      },
    ]
  }
]

useSeoMeta({
  titleTemplate: '',
  title,
  ogTitle: title,
  description,
  ogDescription: description,
  ogImage: 'https://ui.nuxt.com/assets/templates/nuxt/docs-light.png',
  twitterImage: 'https://ui.nuxt.com/assets/templates/nuxt/docs-light.png'
})
</script>

<template>
  <div class="dark:bg-gradient-to-b from-neutral-900 to-neutral-950">
    <UPageHero
      orientation="horizontal"
      class="dark:bg-gradient-to-b from-neutral-900 via-neutral-950 to-neutral-950"
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
        <div class="flex flex-wrap gap-3">
          <UButton
            v-for="(link, index) in hero_links"
            :key="index"
            v-bind="link"
          />
        </div>
      </template>

      <HeroPipelineDiagram />
    </UPageHero>

    <UPageSection
      v-for="(section, index) in sections"
      :key="index"
      class="dark:bg-neutral-950"
      :headline="section.headline"
      :title="section.title"
      :description="section.description"
    >
      <template #links>
        <UButton v-bind="section.link" />
      </template>

      <template #features>
        <UPageFeature
          v-for="(feature, feature_index) in section.features"
          :key="feature_index"
          :icon="feature.icon"
          :title="feature.title"
          :description="feature.description"
        />
      </template>
    </UPageSection>

    <UPageSection class="dark:bg-gradient-to-b from-neutral-950 to-neutral-900">
      <UPageCTA
        title="Inspect the pipeline end to end."
        description="Read the terminology, run a demo, dump the artifacts, and inspect how programs change across lowering and execution boundaries."
        :links="[
          { label: 'Start building', to: '/getting-started', trailingIcon: 'i-lucide-arrow-right' },
          { label: 'Explore the pipeline', to: '/pipeline', variant: 'subtle', trailingIcon: 'i-lucide-arrow-right' },
          { label: 'View on GitHub', to: 'https://github.com/Marco-Christiani/Zigrad', target: '_blank', variant: 'subtle', icon: 'i-simple-icons-github' }
        ]"
        class="dark:bg-neutral-950"
      >
        <StarsBg />
      </UPageCTA>
    </UPageSection>
  </div>
</template>
