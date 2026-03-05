# Zigrad Documentation Site

## Run

```bash
task site:install
task site:dev
```

## Build static site

```bash
task site:build
```

## Routes in this spike

- `/docs` and `/docs/*`: Markdown docs rendered by Nuxt Content
- `/product`, `/demo`, `/contact`: custom Vue pages
- `/autodoc`: embedded Zig autodoc app loading files from `public/api`

## Sync Zig autodoc bundle

The autodoc route expects:

- `public/api/main.js`
- `public/api/main.wasm`
- `public/api/sources.tar`

Sync from generated artifacts:

```bash
./scripts/sync-zig-autodoc.sh /path/to/autodoc-output
```

Or use the integrated Zig build step:

```bash
task docs:web
```

`task docs:web` also applies the project branding patch to `public/api/index.html`.
