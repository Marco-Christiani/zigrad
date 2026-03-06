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

1. Runs `zig build ... docs-web` to emit docs and sync into `public/api`.
2. Applies branding patch from `scripts/brand-zig-autodoc.sh`.
3. Keeps generated artifacts out of git via `.gitignore`.

The `/autodoc` route embeds `public/api/index.html` in an iframe.
