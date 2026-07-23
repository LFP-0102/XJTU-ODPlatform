from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response


class FrontendPagination(PageNumberPagination):
    """分页结构对齐前端 Paginated<T>: { items, total, page, page_size }。"""

    page_size = 10
    page_size_query_param = 'page_size'
    max_page_size = 100

    def get_paginated_response(self, data):
        return Response(
            {
                'items': data,
                'total': self.page.paginator.count,
                'page': self.page.number,
                'page_size': self.get_page_size(self.request),
            }
        )
