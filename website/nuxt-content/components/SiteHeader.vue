<script setup lang="ts">
const color_mode = useColorMode()

function toggle_color_mode(): void {
  color_mode.preference = color_mode.value === "dark" ? "light" : "dark"
}
</script>

<template>
  <header class="site-header">
    <UContainer class="header-inner">
      <NuxtLink class="brand" to="/">Zigrad</NuxtLink>
      <nav class="site-nav">
        <UButton to="/docs" variant="ghost" color="neutral">Docs</UButton>
        <UButton to="/product" variant="ghost" color="neutral">Product</UButton>
        <UButton to="/demo" variant="ghost" color="neutral">Demo</UButton>
        <UButton to="/autodoc" variant="ghost" color="neutral">Autodoc</UButton>
        <UButton to="/contact" variant="ghost" color="neutral">Contact</UButton>
        <UButton variant="soft" color="primary" @click="toggle_color_mode">
          {{ color_mode.value === 'dark' ? 'Light' : 'Dark' }}
        </UButton>
      </nav>
    </UContainer>
  </header>
</template>
