<script setup lang="ts">
const stages = [
  {
    title: 'ZXPR / PR',
    caption: 'Author and validate the program model',
    accent: 'text-emerald-400',
    border: 'border-emerald-500/30',
    bg: 'bg-emerald-500/10'
  },
  {
    title: 'StableHLO / MLIR',
    caption: 'Lower into inspectable compiler artifacts',
    accent: 'text-sky-400',
    border: 'border-sky-500/30',
    bg: 'bg-sky-500/10'
  },
  {
    title: 'PJRT / Providers',
    caption: 'Compile, retarget, and execute',
    accent: 'text-amber-400',
    border: 'border-amber-500/30',
    bg: 'bg-amber-500/10'
  }
]
</script>

<template>
  <div class="w-full max-w-2xl overflow-hidden rounded-2xl border border-white/10 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.12),_transparent_38%),linear-gradient(180deg,rgba(15,23,42,0.96),rgba(3,7,18,0.96))] shadow-2xl">
    <div class="border-b border-white/10 px-5 py-4">
      <div class="flex items-center justify-between gap-4">
        <div>
          <p class="font-mono text-[11px] uppercase tracking-[0.22em] text-cyan-300/80">
            Pipeline Overview
          </p>
          <h3 class="mt-1 text-lg font-semibold text-white">
            One program, multiple compilation stages
          </h3>
        </div>

        <UButton
          to="/pipeline"
          color="neutral"
          variant="outline"
          size="sm"
          trailing-icon="i-lucide-arrow-right"
        >
          Open explorer
        </UButton>
      </div>
    </div>

    <div class="space-y-5 px-5 py-5">
      <div class="grid gap-3 md:grid-cols-[1fr_auto_1fr_auto_1fr] md:items-center">
        <template
          v-for="(stage, index) in stages"
          :key="stage.title"
        >
          <div
            :class="[
              'rounded-xl border p-4',
              stage.border,
              stage.bg
            ]"
          >
            <p
              :class="[
                'font-mono text-xs uppercase tracking-[0.18em]',
                stage.accent
              ]"
            >
              {{ stage.title }}
            </p>
            <p class="mt-2 text-sm text-gray-300">
              {{ stage.caption }}
            </p>
          </div>

          <div
            v-if="index < stages.length - 1"
            class="hidden items-center justify-center text-gray-500 md:flex"
          >
            <span class="font-mono text-xs">lower / compile</span>
          </div>
        </template>
      </div>

      <div class="grid gap-3 sm:grid-cols-3">
        <div class="rounded-lg border border-white/10 bg-white/5 px-4 py-3">
          <p class="font-mono text-[11px] uppercase tracking-[0.18em] text-gray-400">
            Inspect
          </p>
          <p class="mt-1 text-sm text-gray-300">
            Dump PR and MLIR artifacts instead of inferring what the compiler probably did.
          </p>
        </div>

        <div class="rounded-lg border border-white/10 bg-white/5 px-4 py-3">
          <p class="font-mono text-[11px] uppercase tracking-[0.18em] text-gray-400">
            Retarget
          </p>
          <p class="mt-1 text-sm text-gray-300">
            Compare backend-specific lowering without rewriting the source program.
          </p>
        </div>

        <div class="rounded-lg border border-white/10 bg-white/5 px-4 py-3">
          <p class="font-mono text-[11px] uppercase tracking-[0.18em] text-gray-400">
            Optimize
          </p>
          <p class="mt-1 text-sm text-gray-300">
            Hand profitable regions to provider-specific compilation and tuning flows.
          </p>
        </div>
      </div>
    </div>
  </div>
</template>
