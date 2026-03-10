<script setup>
import diagram from "./zigrad-pipeline.svg"
</script>

<template>
  <diagram class="hero-pipeline-diagram" />
</template>

<style>
.hero-pipeline-diagram {
  display: block;
  width: 100%;
  height: auto;

  --diagram-bg: color-mix(in srgb, var(--ui-bg-elevated) 94%, white 6%);
  --diagram-glow: color-mix(in srgb, var(--ui-primary) 18%, transparent);
  --diagram-glow-fade: var(--ui-bg);
  --diagram-border: color-mix(in srgb, var(--ui-border) 72%, transparent);
  --diagram-text-muted: var(--ui-text-toned);
  --diagram-text-header: var(--ui-text-dimmed);

  --diagram-green: #059669;
  --diagram-green-accent: #047857;
  --diagram-blue: #0284c7;
  --diagram-blue-accent: #0369a1;
  --diagram-amber: #d97706;
  --diagram-amber-accent: #b45309;
  --diagram-purple: #a855f7;
  --diagram-purple-accent: #9333ea;
}

:where(.dark, html.dark) .hero-pipeline-diagram {
  --diagram-bg: #06101d;
  --diagram-glow: #155e75;
  --diagram-glow-fade: #030712;
  --diagram-border: rgb(255 255 255 / 0.06);
  --diagram-text-muted: #94a3b8;
  --diagram-text-header: #475569;

  --diagram-green-accent: #34d399;
  --diagram-blue-accent: #38bdf8;
  --diagram-amber-accent: #fbbf24;
  --diagram-purple-accent: #c084fc;
}
</style>
