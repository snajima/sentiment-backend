import json
import datetime

from api.utils import success_response, failure_response
from django.contrib.auth.models import User
from rest_framework import generics
from rest_framework import status

from .serializers import EntrySerializer
from .controllers.create_entry_controller import CreateEntryController
from .controllers.get_date_entries_controller import GetDateEntryController
from .controllers.get_range_entries_controller import GetRangeEntryController
from .controllers.get_person_entries_controller import GetPersonEntryController
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

class DateEntryView(generics.GenericAPIView):
    queryset = Entry.objects
    serializer_class = EntrySerializer

    def post(self, request, id):
        """
        Get entry by user id and date
        """
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = request.data

        return GetDateEntryController(data, request, self.serializer_class, id).process()

    def get(self, request, id):
        """
        Get entry by user id
        """
        return GetPersonEntryController(request, self.serializer_class, id).process()

class RangeEntryView(generics.GenericAPIView):
    queryset = Entry.objects
    serializer_class = EntrySerializer

    def post(self, request, id):
        """
        Get emotion distributions of entries by user id and dates
        """
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = request.data

        return GetRangeEntryController(data, request, self.serializer_class, id).process()