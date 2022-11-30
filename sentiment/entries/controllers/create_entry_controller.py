from api.utils import failure_response
from api.utils import success_response
from django.contrib.auth.models import User
from rest_framework import status
from persons.models import Person

from ..models import Entry
from ..apps import emotion_model

class CreateEntryController:
    def __init__(self, request, data, serializer):
        self._request = request
        self._data = data
        self._serializer = serializer

    def process(self):
        entry_description = self._data.get("entry_description")
        poster_id = self._data.get("poster_id")

        # Verify all required information is provided
        if entry_description is None or poster_id is None:
            return failure_response("Missing entry information", 400)

        poster = Person.objects.filter(id=poster_id)
        if len(poster) != 1:
            return failure_response("Poster does not exist")

        # Create new entry for that user
        entry = Entry.objects.create(
            entry_description=entry_description,
            emotion=emotion_model(entry_description)[0].get('label'),
            poster=poster[0],
        )
        entry.save()

        return success_response(self._serializer(entry).data, status.HTTP_201_CREATED)