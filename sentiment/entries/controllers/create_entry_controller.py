from api.utils import failure_response
from api.utils import success_response
from django.contrib.auth.models import User
from rest_framework import status

from ..models import Entry
from ..apps import emotion_model

class CreateEntryController:
    def __init__(self, request, data, serializer):
        self._request = request
        self._data = data
        self._serializer = serializer

    def process(self):
        entry_description = self._data.get("entry_description")
        user_id = self._data.get("user_id")

        # Verify all required information is provided
        if entry_description is None or user_id is None:
            return failure_response("Missing entry information", 400)

        entry_user = User.objects.get(id=user_id)

        # Create new entry for that user
        entry = Entry.objects.create(
            entry_description=entry_description,
            emotion=emotion_model(entry_description)[0].get('label'),
            user=entry_user,
        )
        entry.save()

        return success_response(self._serializer(entry).data, status.HTTP_201_CREATED)