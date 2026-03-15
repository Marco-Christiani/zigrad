const CONTENT_TYPES: Record<string, string> = {
  '.js': 'application/javascript',
  '.wasm': 'application/wasm',
  '.tar': 'application/x-tar',
  '.html': 'text/html',
  '.svg': 'image/svg+xml'
}

export default defineEventHandler(async (event) => {
  const path = getRouterParam(event, 'path')
  if (!path) {
    throw createError({ statusCode: 400, message: 'Missing path' })
  }

  // In production, serve from R2 binding
  const bucket = event.context.cloudflare?.env?.AUTODOC
  if (bucket) {
    const object = await bucket.get(path)
    if (!object) {
      throw createError({ statusCode: 404, message: `Not found: ${path}` })
    }

    const ext = '.' + path.split('.').pop()
    setHeaders(event, {
      'Content-Type': object.httpMetadata?.contentType || CONTENT_TYPES[ext] || 'application/octet-stream',
      'ETag': object.httpEtag,
      'Cache-Control': 'public, max-age=3600'
    })

    return object.body
  }

  // Local dev fallback: serve from public/api/
  const { readFile } = await import('node:fs/promises')
  const { join } = await import('node:path')
  const filePath = join(process.cwd(), 'public', 'api', path)
  try {
    const data = await readFile(filePath)
    const ext = '.' + path.split('.').pop()
    setHeaders(event, {
      'Content-Type': CONTENT_TYPES[ext] || 'application/octet-stream'
    })
    return data
  }
  catch {
    throw createError({ statusCode: 404, message: `Not found: ${path}` })
  }
})
