from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from apps.recommender.engine import engine

# Create your views here.
class GetRecommendations(APIView):
    def get(self, request, k):
        USER_ID = 111111111
        recommendations = engine(user_id=USER_ID, k=k)

        # LLamar a los modelos y recivir las recomendaciones

        return Response(
            recommendations, 
            status=status.HTTP_200_OK
        )