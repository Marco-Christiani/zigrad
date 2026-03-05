<script setup lang="ts">
defineProps<{
  nav: Array<{ title: string; path?: string; children?: Array<{ title: string; path?: string }> }> | null
}>()
</script>

<template>
  <UCard class="docs-sidebar">
    <template #header>
      <p class="docs-sidebar-title">Documentation</p>
    </template>
    <ul v-if="nav" class="docs-sidebar-list">
      <li v-for="item in nav" :key="item.title">
        <NuxtLink v-if="item.path" :to="item.path">{{ item.title }}</NuxtLink>
        <span v-else>{{ item.title }}</span>
        <ul v-if="item.children?.length">
          <li v-for="child in item.children" :key="child.title">
            <NuxtLink v-if="child.path" :to="child.path">{{ child.title }}</NuxtLink>
            <span v-else>{{ child.title }}</span>
          </li>
        </ul>
      </li>
    </ul>
  </UCard>
</template>
