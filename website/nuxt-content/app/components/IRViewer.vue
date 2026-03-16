<script setup lang="ts">
import cytoscape from 'cytoscape'
// @ts-expect-error cytoscape-elk has no type declarations
import cytoscapeElk from 'cytoscape-elk'
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'

cytoscape.use(cytoscapeElk)

type GraphNode = {
  id: string
  kind: string
  label: string
  vjp?: boolean
  attrs?: Record<string, unknown>
  zxpr?: string
}

type GraphEdge = {
  source: string
  target: string
  port: number
  var: string
  dtype: string
  shape: number[]
}

type GraphRegion = {
  name: string
  kernelize?: string
  outline?: boolean
  node_ids: string[]
  zxpr?: string
}

type GraphFunction = {
  name: string
  nodes: GraphNode[]
  edges: GraphEdge[]
  regions: GraphRegion[]
  returns: string[]
}

type NodeDetail = {
  type: 'node'
  kind: string
  label?: string
  vjp?: boolean
  attrs?: Record<string, unknown>
  zxpr?: string
  region_name?: string
  kernelize?: string | null
  outline?: boolean
}

type EdgeDetail = {
  type: 'edge'
  var_name: string
  dtype: string
  shape: number[]
  port: number
  source: string
  target: string
}

// Op colors are domain-specific (semantic meaning in graph), defined as CSS
// custom properties in the <style> block. This map reads them at runtime so
// Cytoscape nodes get the right fill color.
const OP_COLOR_VARS: Record<string, string> = {
  param: '--irv-op-param',
  add: '--irv-op-elementwise',
  subtract: '--irv-op-elementwise',
  multiply: '--irv-op-elementwise',
  divide: '--irv-op-elementwise',
  maximum: '--irv-op-elementwise',
  exp: '--irv-op-unary',
  log: '--irv-op-unary',
  rsqrt: '--irv-op-unary',
  logistic: '--irv-op-unary',
  convert: '--irv-op-unary',
  dot: '--irv-op-contraction',
  dot_general: '--irv-op-contraction',
  reshape: '--irv-op-shape',
  transpose: '--irv-op-shape',
  broadcast_in_dim: '--irv-op-shape',
  slice: '--irv-op-shape',
  concatenate: '--irv-op-shape',
  iota: '--irv-op-shape',
  reduce_sum: '--irv-op-reduction',
  reduce_max: '--irv-op-reduction',
  compare: '--irv-op-comparison',
  select: '--irv-op-comparison',
  literal: '--irv-op-literal',
  gather: '--irv-op-data',
  scatter: '--irv-op-data',
  call: '--irv-op-call',
  custom_call: '--irv-op-call'
}

const OP_CATEGORIES: [string, string][] = [
  ['Parameter', '--irv-op-param'],
  ['Elementwise', '--irv-op-elementwise'],
  ['Unary', '--irv-op-unary'],
  ['Contraction', '--irv-op-contraction'],
  ['Shape', '--irv-op-shape'],
  ['Reduction', '--irv-op-reduction'],
  ['Comparison', '--irv-op-comparison'],
  ['Literal', '--irv-op-literal'],
  ['Data movement', '--irv-op-data'],
  ['Call', '--irv-op-call']
]

function read_css_var(name: string, fallback?: string): string {
  const val = getComputedStyle(root_el.value!).getPropertyValue(name).trim()
  return val || fallback || '#999'
}

function color_for_kind(kind: string): string {
  const css_var = OP_COLOR_VARS[kind]
  if (css_var) return read_css_var(css_var)
  return read_css_var('--irv-accent')
}

const root_el = ref<HTMLDivElement | null>(null)
const cy_container = ref<HTMLDivElement | null>(null)
const graph_data = ref<GraphFunction[] | null>(null)
const active_function = ref(0)
const selected_detail = ref<NodeDetail | EdgeDetail | null>(null)
const dragging_over = ref(false)
const zxpr_open = ref(true)
const zxpr_width = ref(380)
const resizing = ref(false)

// Layout controls - ELK layered algorithm
const layout_dir = ref<'DOWN' | 'RIGHT'>('DOWN')
const layout_node_sep = ref(20)
const layout_layer_sep = ref(40)

const dir_options = [
  { label: 'Top-Down', value: 'DOWN' as const },
  { label: 'Left-Right', value: 'RIGHT' as const }
]

let cy: cytoscape.Core | null = null

function on_resize_start(event: MouseEvent) {
  event.preventDefault()
  resizing.value = true
  const start_x = event.clientX
  const start_width = zxpr_width.value

  function on_move(e: MouseEvent) {
    const delta = e.clientX - start_x
    zxpr_width.value = Math.max(200, Math.min(800, start_width + delta))
  }

  function on_up() {
    resizing.value = false
    document.removeEventListener('mousemove', on_move)
    document.removeEventListener('mouseup', on_up)
  }

  document.addEventListener('mousemove', on_move)
  document.addEventListener('mouseup', on_up)
}

const SHOW_ALL = -1

const function_options = computed(() => {
  if (!graph_data.value) return []
  const opts = graph_data.value.map((fn, i) => ({ label: fn.name, value: i }))
  if (opts.length > 1) {
    opts.unshift({ label: 'All functions', value: SHOW_ALL })
  }
  return opts
})

const is_all_view = computed(() => active_function.value === SHOW_ALL)

const current_function = computed(() => {
  if (!graph_data.value || is_all_view.value) return null
  return graph_data.value[active_function.value] || null
})

const stat_nodes = computed(() => {
  if (!graph_data.value) return 0
  if (is_all_view.value) return graph_data.value.reduce((s, fn) => s + fn.nodes.length, 0)
  return current_function.value?.nodes.length ?? 0
})
const stat_edges = computed(() => {
  if (!graph_data.value) return 0
  if (is_all_view.value) return graph_data.value.reduce((s, fn) => s + fn.edges.length, 0)
  return current_function.value?.edges.length ?? 0
})
const stat_regions = computed(() => {
  if (!graph_data.value) return 0
  if (is_all_view.value) return graph_data.value.reduce((s, fn) => s + (fn.regions?.length ?? 0), 0)
  return current_function.value?.regions.length ?? 0
})

function zxpr_for_function(fn: GraphFunction): string {
  const params = fn.nodes.filter(n => n.kind === 'param')
  const ops = fn.nodes.filter(n => n.kind !== 'param')
  const lines: string[] = []

  lines.push(`zxpr ${fn.name} {`)
  lines.push(`  ; params (${params.length})`)
  for (const p of params) {
    lines.push(`  ${p.zxpr || p.label}`)
  }
  lines.push(`  ; body (${ops.length} ops)`)
  lines.push('  let')
  for (const op of ops) {
    lines.push(`    ${op.zxpr || op.label}`)
  }
  lines.push(`  in ${fn.returns.join(', ')}`)
  lines.push('}')

  return lines.join('\n')
}

// Reconstruct the full ZXPR text from node data
const function_zxpr = computed(() => {
  if (!graph_data.value) return null
  if (is_all_view.value) {
    return graph_data.value.map(zxpr_for_function).join('\n\n')
  }
  const fn = current_function.value
  return fn ? zxpr_for_function(fn) : null
})

function init_cy() {
  if (!cy_container.value || !root_el.value) return

  const v = read_css_var
  const edge_color = v('--irv-edge')
  const edge_arrow = v('--irv-edge-arrow')
  const selection = v('--irv-selection')
  const region_color = v('--irv-op-param')

  cy = cytoscape({
    container: cy_container.value,
    elements: [],
    layout: { name: 'preset' } as cytoscape.LayoutOptions,
    style: [
      {
        selector: 'node[label]',
        style: {
          'background-color': 'data(color)',
          'label': 'data(label)',
          'color': '#e2e2e2',
          'text-valign': 'center',
          'text-halign': 'center',
          'shape': 'round-rectangle',
          'width': 'label',
          'height': 'label',
          'padding': '8px',
          'font-size': '11px',
          'font-weight': 'bold',
          'font-family': 'monospace'
        }
      },
      {
        selector: 'node[kind = "param"]',
        style: {
          shape: 'ellipse',
          padding: '10px'
        }
      },
      {
        selector: 'edge',
        style: {
          'width': 2,
          'line-color': edge_color,
          'target-arrow-color': edge_arrow,
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier'
        }
      },
      {
        selector: 'edge:selected',
        style: { 'width': 4, 'line-color': selection, 'target-arrow-color': selection }
      },
      {
        selector: 'node:selected',
        style: { 'border-width': 3, 'border-color': selection }
      },
      {
        selector: ':parent',
        style: {
          'background-opacity': 0.1,
          'background-color': region_color,
          'border-width': 1,
          'border-color': region_color,
          'border-opacity': 0.4,
          'label': 'data(regionName)',
          'text-valign': 'top',
          'text-halign': 'center',
          'color': region_color,
          'font-size': '10px',
          'padding': '12px'
        }
      },
      {
        selector: 'node[kind = "function"]',
        style: {
          'background-opacity': 0.06,
          'background-color': v('--irv-accent'),
          'border-width': 2,
          'border-color': v('--irv-accent'),
          'border-opacity': 0.3,
          'border-style': 'dashed',
          'color': v('--irv-accent'),
          'font-size': '12px',
          'font-weight': 'bold',
          'padding': '20px'
        }
      },
      {
        selector: 'edge[crossFunction]',
        style: {
          'line-style': 'dashed',
          'line-dash-pattern': [6, 3],
          'line-color': v('--irv-accent'),
          'target-arrow-color': v('--irv-accent'),
          'width': 2.5
        }
      }
    ] as unknown as cytoscape.StylesheetCSS[]
  })

  cy.on('tap', 'node', (e) => {
    const d = e.target.data()
    if (d.kind === 'region') {
      selected_detail.value = {
        type: 'node',
        kind: 'region',
        region_name: d.name,
        kernelize: d.kernelize,
        outline: d.outline,
        zxpr: d.zxpr
      }
    } else {
      selected_detail.value = {
        type: 'node',
        kind: d.kind,
        label: d.label,
        vjp: d.vjp,
        attrs: d.attrs,
        zxpr: d.zxpr
      }
    }
  })

  cy.on('tap', 'edge', (e) => {
    const d = e.target.data()
    selected_detail.value = {
      type: 'edge',
      var_name: d.varName || '?',
      dtype: d.dtype,
      shape: d.shape,
      port: d.port,
      source: d.source,
      target: d.target
    }
  })

  cy.on('tap', (e) => {
    if (e.target === cy) selected_detail.value = null
  })
}

function render_graph(fn: GraphFunction) {
  if (!cy) return

  const elements: cytoscape.ElementDefinition[] = []

  // Regions as compound nodes
  const region_parent = new Map<string, string>()
  if (fn.regions) {
    for (const [ri, region] of fn.regions.entries()) {
      const region_id = `__region_${ri}`
      elements.push({
        data: {
          id: region_id,
          regionName: `${region.name}${region.kernelize ? ' [' + region.kernelize + ']' : ''}`,
          kind: 'region',
          name: region.name,
          kernelize: region.kernelize || null,
          outline: region.outline || false,
          zxpr: region.zxpr || null
        }
      })
      for (const nid of region.node_ids) {
        region_parent.set(nid, region_id)
      }
    }
  }

  // Nodes
  for (const n of fn.nodes) {
    const node_data: Record<string, unknown> = { ...n }
    const parent = region_parent.get(n.id)
    if (parent) node_data.parent = parent
    node_data.color = color_for_kind(n.kind)
    elements.push({ data: node_data })
  }

  // Edges - label is just the variable name, full detail is in the sidebar on click
  for (const e of fn.edges) {
    elements.push({
      data: {
        source: e.source,
        target: e.target,
        port: e.port,
        varName: e.var,
        dtype: e.dtype,
        shape: e.shape
      }
    })
  }

  cy.elements().remove()
  cy.add(elements)
  run_layout()
  selected_detail.value = null
}

function render_all(fns: GraphFunction[]) {
  if (!cy) return

  const elements: cytoscape.ElementDefinition[] = []
  const fn_map = new Map<string, GraphFunction>()
  for (const fn of fns) fn_map.set(fn.name, fn)

  const ns = (fn_name: string, id: string) => `${fn_name}/${id}`

  for (const fn of fns) {
    // Function as compound parent
    elements.push({
      data: {
        id: `__fn_${fn.name}`,
        regionName: fn.name,
        kind: 'function',
        name: fn.name
      }
    })

    // Regions within function (nested under function compound)
    const region_parent = new Map<string, string>()
    if (fn.regions) {
      for (const [ri, region] of fn.regions.entries()) {
        const region_id = `${fn.name}/__region_${ri}`
        elements.push({
          data: {
            id: region_id,
            parent: `__fn_${fn.name}`,
            regionName: `${region.name}${region.kernelize ? ' [' + region.kernelize + ']' : ''}`,
            kind: 'region',
            name: region.name,
            kernelize: region.kernelize || null,
            outline: region.outline || false,
            zxpr: region.zxpr || null
          }
        })
        for (const nid of region.node_ids) {
          region_parent.set(nid, region_id)
        }
      }
    }

    // Identify call nodes whose callee is in the graph - these get resolved
    const call_nodes = new Map<string, string>()
    for (const n of fn.nodes) {
      if (n.kind === 'call' && n.attrs?.callee && fn_map.has(n.attrs.callee as string)) {
        call_nodes.set(n.id, n.attrs.callee as string)
      }
    }

    // Nodes - skip resolved call nodes
    for (const n of fn.nodes) {
      if (call_nodes.has(n.id)) continue
      const node_data: Record<string, unknown> = { ...n, id: ns(fn.name, n.id) }
      const parent = region_parent.get(n.id)
      node_data.parent = parent || `__fn_${fn.name}`
      node_data.color = color_for_kind(n.kind)
      elements.push({ data: node_data })
    }

    // Edges - resolve cross-function call edges
    for (const e of fn.edges) {
      const target_callee = call_nodes.get(e.target)
      const source_callee = call_nodes.get(e.source)

      if (target_callee) {
        // Edge into call → redirect to callee's param by port index
        const callee = fn_map.get(target_callee)!
        const params = callee.nodes.filter(n => n.kind === 'param')
        if (e.port < params.length) {
          elements.push({
            data: {
              source: ns(fn.name, e.source),
              target: ns(target_callee, params[e.port].id),
              port: e.port,
              varName: e.var,
              dtype: e.dtype,
              shape: e.shape,
              crossFunction: true
            }
          })
        }
      } else if (source_callee) {
        // Edge out of call -> redirect from callee's return by port index
        // returns[] contains var names, not node IDs - resolve via label prefix
        const callee = fn_map.get(source_callee)!
        if (e.port < callee.returns.length) {
          const ret_var = callee.returns[e.port]
          const ret_node = callee.nodes.find(n => n.label.startsWith(ret_var + ' '))
          if (ret_node) {
            elements.push({
              data: {
                source: ns(source_callee, ret_node.id),
                target: ns(fn.name, e.target),
                port: e.port,
                varName: e.var,
                dtype: e.dtype,
                shape: e.shape,
                crossFunction: true
              }
            })
          }
        }
      } else {
        // Normal intra-function edge
        elements.push({
          data: {
            source: ns(fn.name, e.source),
            target: ns(fn.name, e.target),
            port: e.port,
            varName: e.var,
            dtype: e.dtype,
            shape: e.shape
          }
        })
      }
    }
  }

  cy.elements().remove()
  cy.add(elements)
  run_layout()
  selected_detail.value = null
}

function run_layout() {
  if (!cy) return
  cy.layout({
    name: 'elk',
    nodeDimensionsIncludeLabels: true,
    elk: {
      'algorithm': 'layered',
      'elk.direction': layout_dir.value,
      'elk.spacing.nodeNode': String(layout_node_sep.value),
      'elk.layered.spacing.nodeNodeBetweenLayers': String(layout_layer_sep.value),
      'elk.portConstraints': 'FIXED_ORDER',
      'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
      'elk.layered.crossingMinimization.forceNodeModelOrder': 'true'
    }
  } as cytoscape.LayoutOptions).run()
}

function handle_file(file: File) {
  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target?.result as string)
      const arr = Array.isArray(data) ? data : [data]
      graph_data.value = arr
      active_function.value = 0
    } catch (err) {
      console.error('Invalid JSON:', err)
    }
  }
  reader.readAsText(file)
}

function on_file_input(event: Event) {
  const input = event.target as HTMLInputElement
  if (input.files?.[0]) handle_file(input.files[0])
}

function on_drop(event: DragEvent) {
  event.preventDefault()
  dragging_over.value = false
  if (event.dataTransfer?.files[0]) handle_file(event.dataTransfer.files[0])
}

function on_drag_over(event: DragEvent) {
  event.preventDefault()
  dragging_over.value = true
}

function on_drag_leave() {
  dragging_over.value = false
}

function format_attr_value(v: unknown): string {
  if (Array.isArray(v)) return `[${v.join(', ')}]`
  return JSON.stringify(v)
}

function render_current() {
  if (!graph_data.value) return
  if (is_all_view.value) {
    render_all(graph_data.value)
  } else if (current_function.value) {
    render_graph(current_function.value)
  }
}

watch(active_function, render_current)
watch(graph_data, render_current)

// Re-run layout when any layout option changes (no need to rebuild elements)
watch([layout_dir, layout_node_sep, layout_layer_sep], () => {
  if (cy && cy.elements().length > 0) run_layout()
})

onMounted(async () => {
  init_cy()

  try {
    const res = await fetch('/data/graph.json')
    if (res.ok) {
      const data = await res.json()
      const arr = Array.isArray(data) ? data : [data]
      graph_data.value = arr
      active_function.value = 0
    }
  } catch {
    // No default data - user will load manually
  }
})

onUnmounted(() => {
  if (cy) {
    cy.destroy()
    cy = null
  }
})
</script>

<template>
  <div
    ref="root_el"
    class="irv flex flex-col irv-fullscreen"
    :class="{ 'select-none': resizing }"
  >
    <!-- Toolbar -->
    <div class="flex items-center gap-3 border-b border-muted px-4 py-2 shrink-0">
      <USelect
        v-if="function_options.length > 1"
        v-model="active_function"
        :items="function_options"
        value-key="value"
        size="sm"
        class="w-48"
      />
      <span
        v-else-if="current_function"
        class="text-sm font-mono irv-accent"
      >
        {{ current_function.name }}
      </span>

      <!-- Layout controls -->
      <div
        v-if="graph_data"
        class="flex items-center gap-2 ml-4 border-l border-muted pl-4"
      >
        <USelect
          v-model="layout_dir"
          :items="dir_options"
          value-key="value"
          size="xs"
          class="w-28"
        />
        <div class="flex items-center gap-1 text-[10px] font-mono text-dimmed">
          <label>sep</label>
          <input
            v-model.number="layout_node_sep"
            type="range"
            min="5"
            max="80"
            class="irv-range w-16"
          >
          <span class="w-5 text-right text-toned">{{ layout_node_sep }}</span>
        </div>
        <div class="flex items-center gap-1 text-[10px] font-mono text-dimmed">
          <label>layer</label>
          <input
            v-model.number="layout_layer_sep"
            type="range"
            min="10"
            max="120"
            class="irv-range w-16"
          >
          <span class="w-5 text-right text-toned">{{ layout_layer_sep }}</span>
        </div>
      </div>

      <div class="flex-1" />

      <label class="irv-upload-btn cursor-pointer rounded px-3 py-1.5 text-xs font-mono transition-colors">
        Load graph.json
        <input
          type="file"
          accept=".json"
          class="hidden"
          @change="on_file_input"
        >
      </label>

      <div class="flex items-center gap-4 text-xs font-mono text-toned">
        <span>Ops: <span class="irv-accent font-bold">{{ stat_nodes }}</span></span>
        <span>Vars: <span class="irv-accent font-bold">{{ stat_edges }}</span></span>
        <span v-if="stat_regions > 0">Regions: <span class="irv-accent font-bold">{{ stat_regions }}</span></span>
      </div>
    </div>

    <!-- Main area: zxpr panel + graph + detail sidebar -->
    <div class="flex flex-1 min-h-0 min-w-0 overflow-hidden">
      <!-- Left panel: full function ZXPR -->
      <div
        v-if="function_zxpr"
        class="shrink-0 flex flex-col"
        :style="zxpr_open ? { width: zxpr_width + 'px' } : {}"
      >
        <button
          class="flex items-center gap-2 px-3 py-2 text-xs font-mono text-toned hover:text-highlighted transition-colors shrink-0"
          :class="zxpr_open ? 'border-b border-muted' : ''"
          @click="zxpr_open = !zxpr_open"
        >
          <UIcon
            :name="zxpr_open ? 'i-lucide-panel-left-close' : 'i-lucide-panel-left-open'"
            class="size-3.5"
          />
          <template v-if="zxpr_open">
            <span class="irv-accent font-bold">{{ is_all_view ? 'All functions' : current_function?.name }}</span>
          </template>
        </button>
        <div
          v-show="zxpr_open"
          class="overflow-auto flex-1 min-h-0"
        >
          <pre class="irv-zxpr-block px-4 py-3 text-[11px] leading-relaxed whitespace-pre">{{ function_zxpr }}</pre>
        </div>
      </div>

      <!-- Resize handle between ZXPR panel and graph -->
      <div
        v-if="function_zxpr && zxpr_open"
        class="irv-resize-handle shrink-0"
        @mousedown="on_resize_start"
      />

      <!-- Graph canvas -->
      <div
        ref="cy_container"
        class="irv-canvas flex-1 min-w-0 relative"
        @drop="on_drop"
        @dragover="on_drag_over"
        @dragleave="on_drag_leave"
      >
        <!-- Drop overlay -->
        <div
          v-if="dragging_over"
          class="irv-drop-overlay absolute inset-0 z-10 flex items-center justify-center pointer-events-none"
        >
          <span class="irv-accent font-mono text-sm">Drop graph.json here</span>
        </div>

        <!-- Empty state -->
        <div
          v-if="!graph_data"
          class="absolute inset-0 flex items-center justify-center text-muted font-mono text-sm"
        >
          Load a graph.json file or drag and drop
        </div>
      </div>

      <!-- Right sidebar: detail + legend -->
      <div class="w-64 min-w-64 border-l border-muted bg-elevated overflow-y-auto p-4 font-mono text-xs shrink-0">
        <h3 class="irv-accent text-xs font-bold mb-3">
          Detail
        </h3>

        <div
          v-if="!selected_detail"
          class="text-muted"
        >
          Click a node or edge to inspect
        </div>

        <!-- Node detail -->
        <div
          v-else-if="selected_detail.type === 'node'"
          class="space-y-1"
        >
          <template v-if="selected_detail.kind === 'region'">
            <div><span class="text-dimmed">region:</span> <span class="text-highlighted">{{ selected_detail.region_name }}</span></div>
            <div v-if="selected_detail.kernelize">
              <span class="text-dimmed">kernelize:</span> <span class="text-highlighted">{{ selected_detail.kernelize }}</span>
            </div>
            <div v-if="selected_detail.outline">
              <span class="text-dimmed">outline:</span> <span class="text-highlighted">yes</span>
            </div>
          </template>
          <template v-else>
            <div><span class="text-dimmed">op:</span> <span class="text-highlighted">{{ selected_detail.kind }}</span></div>
            <div v-if="selected_detail.label">
              <span class="text-dimmed">label:</span> <span class="text-highlighted">{{ selected_detail.label }}</span>
            </div>
            <div v-if="selected_detail.vjp">
              <span class="text-dimmed">vjp:</span> <span class="text-highlighted">yes</span>
            </div>
            <template v-if="selected_detail.attrs">
              <div class="text-dimmed mt-2">
                attrs:
              </div>
              <div
                v-for="(v, k) in selected_detail.attrs"
                :key="k"
                class="pl-3"
              >
                <span class="text-dimmed">{{ k }}:</span> <span class="text-highlighted">{{ format_attr_value(v) }}</span>
              </div>
            </template>
          </template>
          <div
            v-if="selected_detail.zxpr"
            class="mt-3"
          >
            <span class="text-muted text-[10px]">zxpr</span>
            <pre class="irv-zxpr-block mt-1 p-2 rounded border border-muted text-[11px] leading-relaxed overflow-x-auto whitespace-pre">{{ selected_detail.zxpr }}</pre>
          </div>
        </div>

        <!-- Edge detail -->
        <div
          v-else-if="selected_detail.type === 'edge'"
          class="space-y-1"
        >
          <div><span class="text-dimmed">var:</span> <span class="text-highlighted">{{ selected_detail.var_name }}</span></div>
          <div><span class="text-dimmed">dtype:</span> <span class="irv-dtype">{{ selected_detail.dtype }}</span></div>
          <div><span class="text-dimmed">shape:</span> <span class="irv-accent">[{{ selected_detail.shape?.join(', ') ?? 'scalar' }}]</span></div>
          <div><span class="text-dimmed">port:</span> <span class="text-highlighted">{{ selected_detail.port }}</span></div>
          <div><span class="text-dimmed">source:</span> <span class="text-highlighted">{{ selected_detail.source }}</span></div>
          <div><span class="text-dimmed">target:</span> <span class="text-highlighted">{{ selected_detail.target }}</span></div>
        </div>

        <!-- Legend -->
        <div class="mt-6 pt-4 border-t border-muted">
          <h3 class="irv-accent text-xs font-bold mb-2">
            Op Categories
          </h3>
          <div class="space-y-1">
            <div
              v-for="[name, css_var] in OP_CATEGORIES"
              :key="name"
              class="flex items-center gap-2"
            >
              <span
                class="inline-block w-2.5 h-2.5 rounded-sm"
                :style="{ backgroundColor: `var(${css_var})` }"
              />
              <span class="text-toned">{{ name }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style>
/* IR Viewer theming
   Op colors are domain-specific (semantic graph meaning) so they don't map
   to UI theme tokens. UI chrome uses --ui-* via Tailwind semantic classes.
   Light-mode defaults below; dark overrides follow. */
.irv {
  /* Accent color used for headings, stat values, shape annotations */
  --irv-accent: #0e7490;
  /* Dtype highlight in edge detail */
  --irv-dtype: #9d174d;
  /* ZXPR code block text */
  --irv-zxpr: #92400e;
  /* Graph canvas */
  --irv-canvas-bg: color-mix(in srgb, var(--ui-bg, #fff) 96%, black 4%);
  /* Cytoscape edge styling */
  --irv-edge: var(--ui-border, #d1d5db);
  --irv-edge-arrow: color-mix(in srgb, var(--ui-border, #9ca3af) 80%, black 20%);
  /* Selection highlight */
  --irv-selection: #dc2626;
  /* Drop zone overlay */
  --irv-drop-bg: color-mix(in srgb, var(--irv-accent) 10%, transparent 90%);

  /* Op category colors — light-friendly (darkened for white backgrounds) */
  --irv-op-param: #0369a1;
  --irv-op-elementwise: #4d7c0f;
  --irv-op-unary: #15803d;
  --irv-op-contraction: #b45309;
  --irv-op-shape: #9d174d;
  --irv-op-reduction: #c2410c;
  --irv-op-comparison: #0f766e;
  --irv-op-literal: #57534e;
  --irv-op-data: #a16207;
  --irv-op-call: #dc2626;
}

:where(.dark, html.dark) .irv {
  --irv-accent: #4cc9f0;
  --irv-dtype: #d3869b;
  --irv-zxpr: color-mix(in srgb, var(--ui-warning, #d8a657) 80%, white 20%);
  --irv-canvas-bg: color-mix(in srgb, var(--ui-bg, #000) 92%, black 8%);
  --irv-edge: var(--ui-border, #555);
  --irv-edge-arrow: color-mix(in srgb, var(--ui-border, #777) 80%, white 20%);
  --irv-selection: #ff6b6b;
  --irv-drop-bg: color-mix(in srgb, var(--irv-accent) 15%, transparent 85%);

  /* Op category colors — gruvbox-inspired for dark backgrounds */
  --irv-op-param: #7dcfff;
  --irv-op-elementwise: #a9b665;
  --irv-op-unary: #89b482;
  --irv-op-contraction: #d8a657;
  --irv-op-shape: #d3869b;
  --irv-op-reduction: #e78a4e;
  --irv-op-comparison: #83a598;
  --irv-op-literal: #928374;
  --irv-op-data: #fabd2f;
  --irv-op-call: #fb4934;
}

/* Fill viewport below the site header (UHeader is h-16 = 4rem) */
.irv-fullscreen {
  height: calc(100vh - 4rem);
  overflow: hidden;
  background: var(--ui-bg-elevated);
}

.irv-accent {
  color: var(--irv-accent);
}

.irv-dtype {
  color: var(--irv-dtype);
}

.irv-canvas {
  background: var(--irv-canvas-bg);
}

.irv-upload-btn {
  background: color-mix(in srgb, var(--irv-accent) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--irv-accent) 25%, transparent);
  color: var(--irv-accent);
}

.irv-upload-btn:hover {
  background: color-mix(in srgb, var(--irv-accent) 20%, transparent);
}

.irv-zxpr-block {
  background: color-mix(in srgb, var(--ui-bg) 92%, var(--ui-border) 8%);
  color: var(--irv-zxpr);
}

.irv-resize-handle {
  width: 4px;
  cursor: col-resize;
  background: var(--ui-border);
  transition: background 0.15s;
}

.irv-resize-handle:hover,
.irv-resize-handle:active {
  background: var(--irv-accent);
}

.irv-range {
  -webkit-appearance: none;
  appearance: none;
  height: 3px;
  border-radius: 2px;
  background: var(--ui-border);
  outline: none;
}

.irv-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--irv-accent);
  cursor: pointer;
}

.irv-range::-moz-range-thumb {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--irv-accent);
  border: none;
  cursor: pointer;
}

.irv-drop-overlay {
  background: var(--irv-drop-bg);
  border: 2px dashed var(--irv-accent);
}
</style>
