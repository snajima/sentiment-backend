"""sentiment URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from entries.views import EntriesView, EntryView, DateEntryView
from persons.views import AuthenticateView
from persons.views import DeveloperView
from persons.views import MeView

urlpatterns = [
    path('admin/', admin.site.urls),
    path("entries/", EntriesView.as_view(), name="entries"),
    path("entries/<int:id>/", EntryView.as_view(), name="entry"),
    path("entries/user/<int:id>/", DateEntryView.as_view(), name="entry"),
    path("authenticate/", AuthenticateView.as_view(), name="authenticate"),
    path("dev/", DeveloperView.as_view(), name="dev"),
    path("me/", MeView.as_view(), name="me"),
]
