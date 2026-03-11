export default defineAppConfig({
  ui: {
    colors: {
      primary: 'green',
      neutral: 'slate'
    },
    footer: {
      slots: {
        root: 'border-t border-default',
        left: 'text-sm text-muted'
      }
    }
  },
  seo: {
    siteName: 'Zigrad Docs'
  },
  header: {
    title: 'Zigrad',
    to: '/',
    // logo: {
    //   alt: 'Zigrad logo',
    //   light: '/api/zg-logo.svg',
    //   dark: '/api/zg-logo.svg'
    // },
    search: true,
    colorMode: true,
    links: [
      { label: 'API Reference', to: '/autodoc' },
      { label: 'IR Viewer', to: '/ir-viewer' },
      {
        icon: 'i-simple-icons-github',
        to: 'https://github.com/Marco-Christiani/Zigrad',
        target: '_blank',
        'aria-label': 'GitHub'
      }
    ]
  },
  footer: {
    credits: `Built with Nuxt UI • © ${new Date().getFullYear()} Zigrad`,
    colorMode: false,
    links: [
      {
        icon: 'i-simple-icons-github',
        to: 'https://github.com/Marco-Christiani/Zigrad',
        target: '_blank',
        'aria-label': 'Zigrad on GitHub'
      }
    ]
  },
  toc: {
    title: 'Table of Contents',
    bottom: {
      title: 'Project',
      edit: 'https://github.com/Marco-Christiani/Zigrad/edit/main/website/nuxt-content/content',
      links: [
        {
          icon: 'i-lucide-star',
          label: 'Star on GitHub',
          to: 'https://github.com/Marco-Christiani/Zigrad',
          target: '_blank'
        }
      ]
    }
  }
})
