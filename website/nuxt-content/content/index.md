---
seo:
  title: Zigrad Docs
  description: Documentation for the Zigrad project.
---

::u-page-hero{class="dark:bg-gradient-to-b from-neutral-900 to-neutral-950"}
---
orientation: horizontal
---
#top
:hero-background

#title
Build [Zigrad]{.text-primary} Programs.

#description
Zigrad is a Zig-first gradient and compilation playground for program representation, lowering, and backend execution. Use these docs to learn the pipeline and build end-to-end examples.

#links
  :::u-button
  ---
  to: /getting-started
  size: xl
  trailing-icon: i-lucide-arrow-right
  ---
  Get started
  :::

  :::u-button
  ---
  icon: i-simple-icons-github
  color: neutral
  variant: outline
  size: xl
  to: https://github.com/nuxt-ui-templates/docs
  target: _blank
  ---
  View on GitHub
  :::

#default
  :::prose-pre
  ---
  code: |
    export default defineNuxtConfig({
      modules: [
        '@nuxt/ui',
        '@nuxt/content',
        'nuxt-og-image',
        'nuxt-llms'
      ],

      css: ['~/assets/css/main.css']
    })
  filename: nuxt.config.ts
  ---

  ```ts [nuxt.config.ts]
  export default defineNuxtConfig({
    modules: [
      '@nuxt/ui',
      '@nuxt/content',
      'nuxt-og-image',
      'nuxt-llms'
    ],

    css: ['~/assets/css/main.css']
  })
  ```
  :::
::

::u-page-section{class="dark:bg-neutral-950"}
#title
Core Zigrad Concepts

#links
  :::u-button
  ---
  color: neutral
  size: lg
  target: _blank
  to: https://ui.nuxt.com/docs/getting-started/installation/nuxt
  trailingIcon: i-lucide-arrow-right
  variant: subtle
  ---
  Explore Architecture
  :::

#features
  :::u-page-feature
  ---
  icon: i-lucide-palette
  ---
  #title
  Program Representation (PR)

  #description
  Backend-neutral IR and validation rules that define the canonical representation before lowering.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-type
  ---
  #title
  Lowering to StableHLO/MLIR

  #description
  Transform PR into StableHLO/MLIR artifacts with explicit pass boundaries and inspectable output.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-layers
  ---
  #title
  PJRT Backend Execution

  #description
  Compile and execute lowered artifacts through the PJRT backend with CPU/GPU plugin support.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-search
  ---
  #title
  Pass-Oriented Pipeline

  #description
  Use a pass-centric workflow to produce intermediate artifacts with predictable stage transitions.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-navigation
  ---
  #title
  Reverse-Mode AD (VJP)

  #description
  Track gradients and vector-Jacobian products through the IR to support differentiable workflows.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-moon
  ---
  #title
  Tooling and Diagnostics

  #description
  Inspect PR and MLIR dumps, run focused demos, and iterate quickly with reproducible command flows.
  :::
::

::u-page-section{class="dark:bg-neutral-950"}
#title
Documentation Workflow

#links
  :::u-button
  ---
  color: neutral
  size: lg
  target: _blank
  to: https://content.nuxt.com/docs/getting-started/installation
  trailingIcon: i-lucide-arrow-right
  variant: subtle
  ---
  Open Getting Started
  :::

#features
  :::u-page-feature
  ---
  icon: i-simple-icons-markdown
  ---
  #title
  Getting Started Guides

  #description
  Set up the SDK, configure runtime plugin paths, and run your first demo in minutes.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-file-text
  ---
  #title
  Essentials Reference

  #description
  Learn Markdown features, prose components, code blocks, and structured content patterns used across this site.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-code
  ---
  #title
  Executable Examples

  #description
  Follow concrete Zig and CLI snippets that map directly to repository tasks and runnable flows.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-database
  ---
  #title
  API Reference

  #description
  Browse generated API documentation and source-aligned references for core modules.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-file-code
  ---
  #title
  Design Vocabulary

  #description
  Use consistent stage terminology for transform, lower, compile, and execute across docs and code.
  :::

  :::u-page-feature
  ---
  icon: i-lucide-git-branch
  ---
  #title
  Searchable Knowledge

  #description
  Find architecture notes and practical details quickly with built-in navigation and full-text search.
  :::
::

::u-page-section{class="dark:bg-gradient-to-b from-neutral-950 to-neutral-900"}
  :::u-page-c-t-a
  ---
  links:
    - label: Start building
      to: '/getting-started'
      trailingIcon: i-lucide-arrow-right
    - label: View on GitHub
      to: 'https://github.com/Marco-Christiani/Zigrad'
      target: _blank
      variant: subtle
      icon: i-simple-icons-github
  title: Ready to work with Zigrad?
  description: Start with the essentials, run a demo, and iterate on PR-to-backend flows with confidence.
  class: dark:bg-neutral-950
  ---

  :stars-bg
  :::
::
