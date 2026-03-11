<script setup lang="ts">
useSeoMeta({
  title: 'Pipeline Explorer',
  description: 'Inspect how Zigrad programs move from ZXPR and PR through MLIR lowering into backend execution paths.'
})
</script>

<template>
  <div class="bg-gradient-to-b from-neutral-950 via-neutral-950 to-neutral-900">
    <UPageHero>
      <template #headline>
        Interactive explorer
      </template>

      <template #title>
        Inspect how one program moves through the Zigrad pipeline.
      </template>

      <template #description>
        Compare a stable source representation against backend-specific lowering, then step through a kernel specialization flow.
      </template>

      <template #links>
        <div class="flex flex-wrap gap-3">
          <UButton
            to="/getting-started"
            trailing-icon="i-lucide-arrow-right"
            disabled
          >
            Read the docs
          </UButton>

          <UButton
            to="/"
            color="neutral"
            variant="outline"
          >
            Back to home
          </UButton>
        </div>
      </template>

      <PipelineExplorer />
    </UPageHero>

    <UPageSection
      headline="How to read it"
      title="Use the explorer as a map of stage boundaries."
      description="The main move is conceptual: author once, lower explicitly, then compare how execution paths diverge by backend or provider."
      :features="[
        {
          icon: 'i-lucide-git-compare-arrows',
          title: 'Retarget view',
          description: 'Keep the source program fixed and compare backend-specific lowered output.'
        },
        {
          icon: 'i-lucide-waypoints',
          title: 'Optimization walkthrough',
          description: 'Step through pattern recognition, extraction, tuning, and rewrite without losing source context.'
        },
        {
          icon: 'i-lucide-file-search',
          title: 'Artifact-first debugging',
          description: 'Pair the viewer with real dumped artifacts from the CLI when behavior diverges from expectation.'
        }
      ]"
    />
  </div>
</template>
