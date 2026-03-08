<script setup lang="ts">
import { computed, ref } from 'vue'

type TabValue = 'retarget' | 'optimize'
type Backend = 'xla' | 'iree'

type TabItem = {
  label: string
  value: TabValue
}

type PipelineStep = {
  title: string
  annotation: string
  code: string
  language: string
}

const active_tab = ref<TabValue>('retarget')
const active_backend = ref<Backend>('xla')
const current_step = ref(0)

const tab_items: TabItem[] = [
  { label: 'Deploy Anywhere', value: 'retarget' },
  { label: 'Kernel Fusion', value: 'optimize' }
]

const zxpr_snippet = `entry: train_step
zxpr loss {
  ; params (8)
  a: 784x128<f32>
  b: 128<f32>
  c: 128x64<f32>
  d: 64<f32>
  e: 64x10<f32>
  f: 10<f32>
  g: 64x784<f32>
  h: 64x10<f32>
  ; body (4 ops)
  let
    i: 64x128<f32> = dot[contracting=([1], [0]), K=784](g, a)  ; vjp
    j: 64x128<f32> = broadcast_in_dim[[128] -> [64, 128], dims=[1]](b)  ; vjp
    k: 64x128<f32> = add(i, j)  ; vjp
    l: f32 = reduce_sum[axes=[0, 1]](k)  ; vjp
  in l
}

zxpr train_step {
  ; params (8)
  a: 784x128<f32>
  b: 128<f32>
  c: 128x64<f32>
  d: 64<f32>
  e: 64x10<f32>
  f: 10<f32>
  g: 64x784<f32>
  h: 64x10<f32>
  ; body (3 ops)
  let
    i: f32, j: 784x128<f32>, k: 128<f32> = call[callee="loss_vjp"](a, b, c, d, e, f, g, h)
    l: f32 = literal[0.01]()
    m: 784x128<f32> = broadcast_in_dim[[] -> [784, 128], dims=[]](l)  ; vjp
  in i, j, k
}`

const xla_snippet = `// Lowered: StableHLO -> XLA/PJRT
func.func @matmul_relu(%arg0: tensor<512x512xf32>,
                       %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1,
       contracting_dims = [1] x [0] : (tensor<512x512xf32>,
       tensor<512x512xf32>) -> tensor<512x512xf32>
  %1 = stablehlo.maximum %0, zeroes : tensor<512x512xf32>
  return %1
}`

const iree_snippet = `// Lowered: IREE flow dialect
flow.executable @matmul_relu_dispatch {
  flow.executable.export @matmul_relu_dispatch_0
  builtin.module {
    func.func @matmul_relu_dispatch_0(
        %arg0: !flow.dispatch.tensor<readonly:tensor<512x512xf32>>,
        %arg1: !flow.dispatch.tensor<readonly:tensor<512x512xf32>>,
        %out:  !flow.dispatch.tensor<writeonly:tensor<512x512xf32>>) {
      // tiled matmul + fused relu
    }
  }
}`

const pipeline_steps: PipelineStep[] = [
  {
    title: 'Source PR',
    annotation: 'Unoptimized program representation',
    language: 'text',
    code: `func @attention(%Q: tensor<64x128xf32>,
              %K: tensor<64x128xf32>,
              %V: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %scores = zg.matmul(%Q, zg.transpose(%K))
  %scaled = zg.div(%scores, zg.const(11.31))
  %weights = zg.softmax(%scaled, axis=-1)
  %out = zg.matmul(%weights, %V)
  return %out
}`
  },
  {
    title: 'Pattern Match',
    annotation: 'Scaled dot-product attention pattern recognized',
    language: 'text',
    code: `func @attention(%Q: tensor<64x128xf32>,
              %K: tensor<64x128xf32>,
              %V: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // region matched: scaled_dot_product_attention
  %scores = zg.matmul(%Q, zg.transpose(%K))
  %scaled = zg.div(%scores, zg.const(11.31))
  %weights = zg.softmax(%scaled, axis=-1)
  %out = zg.matmul(%weights, %V)
  return %out
}`
  },
  {
    title: 'Extract + Tune',
    annotation: 'Region sent to TVM/Mirage for specialized compilation and autotuning',
    language: 'text',
    code: `// Region extracted -> provider compilation
// Provider: TVM
// Target: CUDA sm_90
// Status: autotuning (1024 trials)
// Best: 0.42ms (vs 0.89ms native)
// Speedup: 2.1x - PROFITABLE`
  },
  {
    title: 'Rewrite',
    annotation: 'Profitable region replaced with custom_call dispatch',
    language: 'mlir',
    code: `func @attention(%Q: tensor<64x128xf32>,
              %K: tensor<64x128xf32>,
              %V: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %out = stablehlo.custom_call
           @zigrad.provider.dispatch(%Q, %K, %V) {
    provider = "tvm",
    kernel   = "sdpa_fused_sm90_v3",
    artifact = "sdpa_fused_sm90_v3.so"
  } : (tensor<64x128xf32>, tensor<64x128xf32>,
       tensor<64x128xf32>) -> tensor<64x128xf32>
  return %out
}`
  }
]

const fallback_pipeline_step: PipelineStep = {
  title: '',
  annotation: '',
  code: '',
  language: 'text'
}

const backend_snippet = computed(() => active_backend.value === 'xla' ? xla_snippet : iree_snippet)
const current_pipeline = computed<PipelineStep>(() => pipeline_steps[current_step.value] ?? fallback_pipeline_step)
const can_go_back = computed(() => current_step.value > 0)
const can_go_forward = computed(() => current_step.value < pipeline_steps.length - 1)

function next_step(): void {
  if (can_go_forward.value) current_step.value += 1
}

function prev_step(): void {
  if (can_go_back.value) current_step.value -= 1
}
</script>

<template>
  <div class="w-full max-w-2xl overflow-hidden rounded-xl border border-gray-800 bg-gray-950 shadow-2xl">
    <UTabs
      v-model="active_tab"
      :items="tab_items"
      variant="link"
      class="border-b border-gray-800"
    />

    <div
      v-if="active_tab === 'retarget'"
      class="space-y-3 p-4"
    >
      <div>
        <div class="mb-1.5 flex items-center gap-2">
          <span class="font-mono text-xs text-emerald-400">Program Representation</span>
          <span class="font-mono text-[10px] text-gray-500">// same for all backends</span>
        </div>
        <CodeBlock
          language="zxpr"
          filename="program.zxpr"
          :code="zxpr_snippet"
        />
      </div>

      <div class="flex items-center gap-2 text-gray-500">
        <div class="flex-1 border-t border-gray-800" />
        <span class="font-mono text-xs">lower</span>
        <div class="flex-1 border-t border-gray-800" />
      </div>

      <div>
        <div class="mb-1.5 flex items-center gap-2">
          <button
            :class="[
              'rounded px-2 py-0.5 font-mono text-xs transition-colors',
              active_backend === 'xla' ? 'bg-blue-500/20 text-blue-400' : 'text-gray-500 hover:text-gray-300'
            ]"
            @click="active_backend = 'xla'"
          >
            XLA / PJRT
          </button>
          <button
            :class="[
              'rounded px-2 py-0.5 font-mono text-xs transition-colors',
              active_backend === 'iree' ? 'bg-amber-500/20 text-amber-400' : 'text-gray-500 hover:text-gray-300'
            ]"
            @click="active_backend = 'iree'"
          >
            IREE
          </button>
        </div>
        <MagicCodeBlock
          language="mlir"
          :filename="active_backend === 'xla' ? 'lowered_xla.mlir' : 'lowered_iree.mlir'"
          :code="backend_snippet"
        />
      </div>
    </div>

    <div
      v-else
      class="space-y-3 p-4"
    >
      <div class="flex items-center justify-between">
        <div>
          <span class="font-mono text-xs text-purple-400">
            {{ current_pipeline.title }}
          </span>
          <span class="ml-2 font-mono text-[10px] text-gray-500">
            {{ current_step + 1 }}/{{ pipeline_steps.length }}
          </span>
        </div>

        <div class="flex gap-1">
          <div
            v-for="(_, idx) in pipeline_steps"
            :key="idx"
            :class="[
              'h-1.5 w-1.5 rounded-full transition-colors',
              idx === current_step ? 'bg-purple-400' : 'bg-gray-700'
            ]"
          />
        </div>
      </div>

      <p class="font-mono text-[11px] text-gray-400">
        {{ current_pipeline.annotation }}
      </p>

      <div class="min-h-[220px]">
        <MagicCodeBlock
          :language="current_pipeline.language"
          :filename="`pipeline_step_${current_step + 1}.ir`"
          :code="current_pipeline.code"
        />
      </div>

      <div class="flex items-center justify-between">
        <button
          :disabled="!can_go_back"
          :class="[
            'rounded px-3 py-1 font-mono text-xs transition-colors',
            can_go_back ? 'text-gray-300 hover:bg-gray-800' : 'cursor-not-allowed text-gray-700'
          ]"
          @click="prev_step"
        >
          prev
        </button>

        <button
          :disabled="!can_go_forward"
          :class="[
            'rounded px-3 py-1 font-mono text-xs transition-colors',
            can_go_forward ? 'text-gray-300 hover:bg-gray-800' : 'cursor-not-allowed text-gray-700'
          ]"
          @click="next_step"
        >
          next
        </button>
      </div>
    </div>
  </div>
</template>
