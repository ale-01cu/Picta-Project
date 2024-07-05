from apps.pubs.views import ListPubs, GetPubById
from django.urls import path

urlpatterns = [
    path('pubs/', ListPubs.as_view()),
    path('pubs/<int:pub_id>/', GetPubById.as_view()),
]
