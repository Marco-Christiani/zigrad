<script setup lang="ts">
const route = useRoute()

const normalizedPath = computed(() => {
  const path = route.path.replace(/\/$/, "")
  return path.length == 0 ? "/docs" : path
})

const { data: page } = await useAsyncData(
  () => `docs-page:${normalizedPath.value}`,
  () => queryCollection("docs").path(normalizedPath.value).first(),
  { watch: [normalizedPath] }
)

const { data: nav } = await useAsyncData("docs-nav", () => queryCollectionNavigation("docs"))
</script>

<template>
  <div class="docs-layout">
    <DocsSidebar :nav="nav" />
    <UCard class="docs-article">
      <div v-if="page" class="prose prose-neutral dark:prose-invert max-w-none">
        <ContentRenderer :value="page" />
      </div>
      <div v-else>
        <h1>Not Found</h1>
        <p>No documentation page exists at this route.</p>
      </div>
    </UCard>
  </div>
</template>
