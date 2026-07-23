"""
Django settings — od_web_backend
脑部 MRI 肿瘤检测系统后端。复用同仓 od_platform 引擎(推理 / 可视化 / 模型解析)。
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 路径:定位仓库 workspace 根(.odp-workspace 标记),把引擎 src 挂上 sys.path
# ---------------------------------------------------------------------------
# BASE_DIR = apps/web-backend
BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / '.env')


def _find_workspace_root(start: Path, marker: str = '.odp-workspace') -> Path:
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    # 兜底:apps/web-backend 上两级通常就是仓库根
    return start.parents[1]


WORKSPACE_ROOT = _find_workspace_root(BASE_DIR)
ENGINE_SRC = WORKSPACE_ROOT / 'apps' / 'platform' / 'src'
if ENGINE_SRC.exists() and str(ENGINE_SRC) not in sys.path:
    # 让后端能 `import od_platform`(生产建议改为 `pip install -e apps/platform`)
    sys.path.insert(0, str(ENGINE_SRC))

# ---------------------------------------------------------------------------
# 基本
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'dev-insecure-change-me-in-prod')
DEBUG = os.getenv('DJANGO_DEBUG', 'true').lower() != 'false'
ALLOWED_HOSTS = os.getenv('DJANGO_ALLOWED_HOSTS', '*').split(',')

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'od_web_backend.core',
    'od_web_backend.inference',
    'od_web_backend.history',
    'od_web_backend.analysis',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'od_web_backend.urls'
WSGI_APPLICATION = 'od_web_backend.wsgi.application'
ASGI_APPLICATION = 'od_web_backend.asgi.application'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'src' / 'od_web_backend' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {'context_processors': []},
    },
]

# ---------------------------------------------------------------------------
# 数据库:开发 SQLite,生产可用环境变量切 PostgreSQL
# ---------------------------------------------------------------------------
if os.getenv('POSTGRES_DB'):
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.getenv('POSTGRES_DB'),
            'USER': os.getenv('POSTGRES_USER', 'postgres'),
            'PASSWORD': os.getenv('POSTGRES_PASSWORD', ''),
            'HOST': os.getenv('POSTGRES_HOST', '127.0.0.1'),
            'PORT': os.getenv('POSTGRES_PORT', '5432'),
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ---------------------------------------------------------------------------
# 媒体(端私有:上传原图 / 检测图 / 报告)
# ---------------------------------------------------------------------------
MEDIA_URL = '/media/'
MEDIA_ROOT = Path(os.getenv('MEDIA_ROOT', BASE_DIR / 'media'))
STATIC_URL = '/static/'

LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# DRF:统一信封渲染 + 异常处理
# ---------------------------------------------------------------------------
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'od_web_backend.core.renderers.EnvelopeJSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    'EXCEPTION_HANDLER': 'od_web_backend.core.exceptions.envelope_exception_handler',
    # 关闭 DRF 的 ?format= 内容协商覆盖:report?format=pdf|docx 是业务参数,不是渲染器格式
    'URL_FORMAT_OVERRIDE': None,
    'UNAUTHENTICATED_USER': None,
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': [],
}

# CORS:开发放开(生产由 Nginx 同源或按需白名单)
CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOWED_ORIGINS = [
    o for o in os.getenv('CORS_ALLOWED_ORIGINS', '').split(',') if o
]

# ---------------------------------------------------------------------------
# 业务:推理后端 / 模型缓存 / 可视化 / 大模型 / 报告
# ---------------------------------------------------------------------------
# 推理后端:'yolo' = 真实 ultralytics 推理(生产默认);
#           'demo' = 免 torch,用真实 BeautifyVisualizer 画合成检测(无模型/无 GPU 时演示)
INFER_BACKEND = os.getenv('INFER_BACKEND', 'yolo').lower()
INFER_DEVICE = os.getenv('INFER_DEVICE', 'cpu')  # 'cpu' / 'cuda:0' ...
MODEL_CACHE_SIZE = int(os.getenv('MODEL_CACHE_SIZE', '2'))  # LRU 缓存的模型数

# 可视化:类别中文映射 + 颜色(BGR)+ 字体
VIZ_USE_LABEL_MAPPING = os.getenv('VIZ_USE_LABEL_MAPPING', 'true').lower() != 'false'
VIZ_LABEL_MAPPING = {
    'glioma': '胶质瘤',
    'meningioma': '脑膜瘤',
    'pituitary': '垂体瘤',
}
VIZ_COLOR_MAPPING = {  # BGR(与前端 --cls-* 语义一致)
    'glioma': (68, 68, 239),      # 红
    'meningioma': (11, 158, 245),  # 琥珀
    'pituitary': (246, 92, 139),   # 紫
}
# 字体:引擎默认用 visualization/assets 内置字体;此处可覆盖为系统 CJK 字体绝对路径
VIZ_FONT_PATH = os.getenv('VIZ_FONT_PATH') or None

# 演示模式下对外呈现的模型(真实模式改从 models/trained 扫描)
DEMO_MODELS = [
    {
        'name': 'brain-tumor-yolo11s-best.pt',
        'task': 'detect',
        'classes': ['glioma', 'meningioma', 'pituitary'],
        'metrics': {'mAP50': 0.912, 'mAP50-95': 0.678, 'precision': 0.9, 'recall': 0.87},
    },
    {
        'name': 'brain-tumor-yolo11m-best.pt',
        'task': 'detect',
        'classes': ['glioma', 'meningioma', 'pituitary'],
        'metrics': {'mAP50': 0.934, 'mAP50-95': 0.712, 'precision': 0.921, 'recall': 0.895},
    },
]

# 大模型:'template' = 无需密钥的结构化模板分析(默认,开箱即用);
#         'dashscope' = 通义千问;'openai' = OpenAI。缺密钥/SDK 时自动回退 template。
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'template').lower()
LLM_MODEL = os.getenv('LLM_MODEL', 'qwen-plus')
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', '')

# 报告 PDF 引擎:'auto'(优先 WeasyPrint,缺系统库回退 xhtml2pdf)/ 'weasyprint' / 'xhtml2pdf'
REPORT_PDF_ENGINE = os.getenv('REPORT_PDF_ENGINE', 'auto').lower()

# 上传限制
MAX_BATCH_IMAGES = int(os.getenv('MAX_BATCH_IMAGES', '60'))
DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024  # 100MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 20 * 1024 * 1024
