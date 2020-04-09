from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='Home'),
    path('about/', views.about, name='About'),
    path('summarise/', views.summarise, name='Summarise'),
    path('summary/', views.summary, name='Summary'),
]
