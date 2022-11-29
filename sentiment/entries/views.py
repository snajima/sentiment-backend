import json

from api.utils import success_response, failure_response
from django.contrib.auth.models import User
from rest_framework import generics
from rest_framework import status

from .serializers import EntrySerializer
from .controllers.create_entry_controller import CreateEntryController
from .controllers.update_entry_controller import UpdateEntryController
from .models import Entry

class EntriesView(generics.GenericAPIView):
    queryset = Entry.objects
    serializer_class = EntrySerializer

    def get(self, request):
        """
        Get entry
        """
        return success_response({"entries": self.serializer_class(self.queryset.all(), many=True).data})

    def post(self, request):
        """
        Create entry
        """
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = request.data
        return CreateEntryController(request, data, self.serializer_class).process()

class EntryView(generics.GenericAPIView):
    queryset = Entry.objects
    serializer_class = EntrySerializer

    def get(self, request, id):
        """
        Get entry by id
        """
        if not self.queryset.filter(id=id).exists():
            return failure_response("Entry does not exist")
        entry = self.queryset.get(id=id)
        return success_response(self.serializer_class(entry).data, status.HTTP_200_OK)

    def post(self, request, id):
        """
        Update entry by id
        """
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = request.data
        return UpdateEntryController(request, data, self.serializer_class, id).process()