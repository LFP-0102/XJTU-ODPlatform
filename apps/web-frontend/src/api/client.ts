import axios, { type AxiosRequestConfig } from 'axios'
import { ElMessage } from 'element-plus'
import type { ApiEnvelope } from '@/types'

export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false'

const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '/api',
  timeout: 120_000,
})

// 请求拦截:预留鉴权 token 注入位
http.interceptors.request.use((config) => {
  const token = localStorage.getItem('od_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// 响应拦截:拆 { code, message, data } 信封,非 0 统一报错
http.interceptors.response.use(
  (resp) => {
    // 二进制(报告下载等)直接透传
    if (resp.config.responseType === 'blob') return resp
    const body = resp.data as ApiEnvelope<unknown>
    if (body && typeof body === 'object' && 'code' in body) {
      if (body.code !== 0) {
        ElMessage.error(body.message || '请求失败')
        return Promise.reject(new Error(body.message))
      }
      return { ...resp, data: body.data }
    }
    return resp
  },
  (error) => {
    const msg =
      error?.response?.data?.message || error?.message || '网络异常,请稍后重试'
    ElMessage.error(msg)
    return Promise.reject(error)
  }
)

/** 真实请求 —— 返回已拆信封后的 data 部分 */
export async function request<T>(config: AxiosRequestConfig): Promise<T> {
  const resp = await http.request<T>(config)
  return resp.data as T
}

export { http }
