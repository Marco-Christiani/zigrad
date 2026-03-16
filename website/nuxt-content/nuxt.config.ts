import svgLoader from "vite-svg-loader"

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  vite: {
    plugins: [svgLoader()]
  },
  modules: [
    '@nuxt/eslint',
    '@nuxt/image',
    '@nuxt/ui',
    '@nuxt/content',
    'nuxt-og-image',
    'nuxt-llms',
    '@nuxtjs/mcp-toolkit',
    'nuxt-studio'
  ],

  devtools: {
    enabled: true
  },

  css: ['~/assets/css/main.css', 'shiki-magic-move/style.css'],

  content: {
    build: {
      markdown: {
        toc: {
          searchDepth: 1
        }
      }
    }
  },

  mdc: {
    highlight: {
      noApiRoute: true,
      langs: ['js', 'jsx', 'json', 'ts', 'tsx', 'vue', 'css', 'html', 'bash', 'md', 'mdc', 'yaml', 'zig', 'llvm']
    }
  },

  experimental: {
    asyncContext: true
  },

  app: {
    head: {
      link: [
        { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }
      ]
    }
  },

  compatibilityDate: '2024-07-11',

  routeRules: {
    '/autodoc': { ssr: false }
  },

  nitro: {
    prerender: {
      routes: [
        '/'
      ],
      crawlLinks: true,
      autoSubfolderIndex: false,
      ignore: ['/autodoc']
    }
  },

  eslint: {
    config: {
      stylistic: {
        commaDangle: 'never',
        braceStyle: '1tbs'
      }
    }
  },

  icon: {
    provider: 'iconify'
  },

  llms: {
    domain: 'https://docs-template.nuxt.dev/',
    title: 'Zigrad Docs',
    description: 'Documentation for the Zigrad project.',
    full: {
      title: 'Zigrad Docs - Full Documentation',
      description: 'This is the full documentation for the Zigrad project.'
    },
    sections: [
      {
        title: 'Getting Started',
        contentCollection: 'docs',
        contentFilters: [
          { field: 'path', operator: 'LIKE', value: '/getting-started%' }
        ]
      },
      {
        title: 'Essentials',
        contentCollection: 'docs',
        contentFilters: [
          { field: 'path', operator: 'LIKE', value: '/essentials%' }
        ]
      }
    ]
  },

  studio: {
    repository: {
      provider: 'github',
      owner: 'Marco-Christiani',
      repo: 'zigrad',
      branch: 'modular'
    }
  },

  mcp: {
    name: 'Zigrad Docs'
  }
})
