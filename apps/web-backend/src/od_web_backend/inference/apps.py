from django.apps import AppConfig


class InferenceConfig(AppConfig):
    name = 'od_web_backend.inference'
    label = 'inference'

    def ready(self):
        # 关掉 OpenCV 内部多线程:避免在 Django 开发服务器(多线程 WSGI)的
        # 工作线程里调用 cv2 时发生原生崩溃。对生产 gunicorn(多进程)无副作用。
        try:
            import cv2
            cv2.setNumThreads(0)
        except Exception:
            pass
