from rest_framework.renderers import JSONRenderer


class EnvelopeJSONRenderer(JSONRenderer):
    """统一响应信封:所有 JSON 响应包成 { code, message, data }。

    - 视图直接返回业务数据(dict / list)→ 包成 { code: 0, message: 'ok', data: <数据> }
    - 已经是信封结构(异常处理器产出的 { code, message, data })→ 原样透传
    """

    def render(self, data, accepted_media_type=None, renderer_context=None):
        if not (isinstance(data, dict) and 'code' in data and 'message' in data):
            data = {'code': 0, 'message': 'ok', 'data': data}
        return super().render(data, accepted_media_type, renderer_context)
