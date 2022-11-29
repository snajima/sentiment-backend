from api.utils import failure_response
from api.utils import success_response
from api.utils import update
from django.contrib.auth.models import User

from ..models import Entry
from ..apps import emotion_model

class UpdateEntryController:
    def __init__(self, request, data, serializer, id):
        self._request = request
        self._data = data
        self._serializer = serializer
        self._id = id

    def process(self):
        if not Entry.objects.filter(id=int(self._id)).exists():
            return failure_response("Entry does not exist")
        entry = Entry.objects.get(id=self._id)

        # Extract attributes
        entry_description = self._data.get("entry_description")
        emotion = self._data.get("emotion")
        
        if entry_description is not None:
            update(entry, "entry_description", entry_description)
            update(entry, "emotion", emotion_model(entry_description)[0].get('label'))
        update(entry, "emotion", emotion)

        # Save new changes
        entry.save()
        entry = Entry.objects.get(id=entry.id)
        return success_response(self._serializer(entry).data)