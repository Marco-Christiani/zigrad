# Zigrad Website (Nuxt UI + Nuxt Content)

Template-based docs site with Zigrad-specific routes and autodoc embedding.

## Workflow

```bash
task site:install
task docs:web
task site:dev
```

## Build static output

```bash
task site:build
```

## Zig autodoc integration

`task docs:web` does three things:

1. Runs `zig build ... docs` to emit the Zig autodoc bundle.
2. Syncs the runtime autodoc assets into `public/api`.
3. Leaves the Nuxt-side presentation and styling to `app/components/ZigAutodoc.vue`.

The `/autodoc` route mounts the autodoc viewer in-page and loads `public/api/main.js`, `main.wasm`, and `sources.tar` directly.
