export default defineNuxtConfig({
  compatibilityDate: "2026-03-05",
  modules: ["@nuxt/content", "@nuxt/ui", "@nuxtjs/color-mode"],
  css: ["~/assets/css/main.css"],
  devtools: { enabled: true },
  colorMode: {
    preference: "system",
    fallback: "light",
    classSuffix: ""
  },
  app: {
    head: {
      title: "Zigrad",
      meta: [
        {
          name: "description",
          content: "High-performance deep learning framework in Zig"
        }
      ]
    }
  },
  nitro: {
    prerender: {
      routes: ["/", "/docs", "/product", "/demo", "/contact", "/autodoc"]
    }
  }
})
