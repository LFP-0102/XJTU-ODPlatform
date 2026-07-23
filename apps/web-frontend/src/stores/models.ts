import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { DetectionModel } from '@/types'
import { listModels, syncModels } from '@/api/models'

export const useModelsStore = defineStore('models', () => {
  const models = ref<DetectionModel[]>([])
  const loading = ref(false)
  const loaded = ref(false)

  async function load(force = false) {
    if (loaded.value && !force) return
    loading.value = true
    try {
      models.value = await listModels()
      loaded.value = true
    } finally {
      loading.value = false
    }
  }

  async function sync() {
    loading.value = true
    try {
      models.value = await syncModels()
      loaded.value = true
    } finally {
      loading.value = false
    }
  }

  function byName(name: string) {
    return models.value.find((m) => m.name === name)
  }

  return { models, loading, loaded, load, sync, byName }
})
