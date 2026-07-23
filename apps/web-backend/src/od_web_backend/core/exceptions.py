import logging

from rest_framework.views import exception_handler as drf_exception_handler
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger(__name__)


def _extract_message(data) -> str:
    """把 DRF 的错误结构压成一句人类可读的话。"""
    if isinstance(data, dict):
        if 'detail' in data:
            return str(data['detail'])
        # 字段级错误:取第一个
        for key, val in data.items():
            if isinstance(val, (list, tuple)) and val:
                return f'{key}: {val[0]}'
            return f'{key}: {val}'
    if isinstance(data, list) and data:
        return str(data[0])
    return str(data)


def envelope_exception_handler(exc, context):
    """把所有异常统一成 { code, message, data: null }(HTTP 状态码保持语义)。"""
    response = drf_exception_handler(exc, context)

    if response is not None:
        message = _extract_message(response.data)
        response.data = {'code': response.status_code, 'message': message, 'data': None}
        return response

    # DRF 没接住的异常(未预期错误)
    logger.exception('未处理异常: %s', exc)
    return Response(
        {'code': 500, 'message': f'服务器内部错误: {exc}', 'data': None},
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
