from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from rest_framework import views
from rest_framework.views import Response
from rest_framework.request import Request
from apps.pubs.models import PubsModel, ClicksModel
from rest_framework import status
from datetime import date, datetime
from database.utils import get_id


class ListPubs(views.APIView):
    def get(self, request: Request):
        query = request.query_params.get('page')
        try:
            query = int(query)

        except:
            query = 1
            
        page = query if query and int(query) > 0 else 1
        pubs = PubsModel().list_all(page=page)
        
        data = {
            "previous": page-1 if page>0 else 0,
            "next": page+1,
            "pubs": pubs
        }
        
        return Response(
            data=data, 
            status=status.HTTP_200_OK
        )
    

class GetPubById(views.APIView):
    def get(self, request: Request, pub_id: int):
        USER_ID = 111111111
        click_id = get_id()
        clicks_model = ClicksModel()
        if clicks_model.get_by_id(click_id):
            pass
        else:
            clicks_model.create(data=[
                click_id, datetime.now(), USER_ID, pub_id
            ])

        pub = PubsModel().get_by_id(pub_id)
        return Response(pub, status=status.HTTP_200_OK)

