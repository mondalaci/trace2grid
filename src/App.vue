<script setup lang="ts">
import { computed } from 'vue'
import { useProjectStore } from './stores/project'
import type { Step } from './types'
import CaptureView from './views/CaptureView.vue'
import EditView from './views/EditView.vue'
import PreviewView from './views/PreviewView.vue'

const store = useProjectStore()

const steps: { id: Step; label: string; hint: string }[] = [
  { id: 'capture', label: '1. Scan', hint: 'Photo of tools on paper' },
  { id: 'edit', label: '2. Arrange', hint: 'Position, notches, bin size' },
  { id: 'preview', label: '3. Print', hint: '3D preview & export' },
]

const currentView = computed(() => {
  switch (store.step) {
    case 'capture':
      return CaptureView
    case 'edit':
      return EditView
    case 'preview':
      return PreviewView
  }
})

function goTo(step: Step) {
  if (step !== 'capture' && store.tools.length === 0) return
  store.step = step
}
</script>

<template>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <span class="logo">⬛</span>
        <h1>Trace2Grid</h1>
        <span class="tagline muted">photo → Gridfinity bin</span>
      </div>
      <nav class="steps">
        <button
          v-for="s in steps"
          :key="s.id"
          class="step"
          :class="{ active: store.step === s.id }"
          :disabled="s.id !== 'capture' && store.tools.length === 0"
          :title="s.hint"
          @click="goTo(s.id)"
        >
          {{ s.label }}
        </button>
      </nav>
    </header>
    <main class="content">
      <component :is="currentView" />
    </main>
  </div>
</template>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 18px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-raised);
  flex-wrap: wrap;
}

.brand {
  display: flex;
  align-items: baseline;
  gap: 10px;
}

.logo {
  color: var(--accent);
  font-size: 16px;
}

.brand h1 {
  font-size: 17px;
}

.tagline {
  font-size: 12.5px;
}

.steps {
  display: flex;
  gap: 6px;
}

.step {
  border-radius: 999px;
  padding: 6px 16px;
}

.step.active {
  background: var(--accent-soft);
  border-color: var(--accent);
  color: var(--accent);
  font-weight: 600;
}

.content {
  flex: 1;
  min-height: 0;
  overflow: auto;
}
</style>
