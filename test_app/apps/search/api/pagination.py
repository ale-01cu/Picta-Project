from rest_framework.pagination import PageNumberPagination

class ContenidoPagination(PageNumberPagination):
  page_size = 10
  page_query_param = 'p'