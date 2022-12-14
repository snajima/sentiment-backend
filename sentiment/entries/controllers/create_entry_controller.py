import datetime

from api.utils import failure_response
from api.utils import success_response
from api.utils import update
from django.contrib.auth.models import User
from rest_framework import status
from persons.models import Person

from ..models import Entry
from .algorithm.algorithm import top_labels

class CreateEntryController:
    def __init__(self, request, data, serializer):
        self._request = request
        self._data = data
        self._serializer = serializer

    def process(self):
        entry_description = self._data.get("entry_description")
        emotion = self._data.get("emotion")
        poster_id = self._data.get("poster_id")

        # convert date to datetime object
        try:
            date = datetime.datetime.strptime(self._data.get("date"), "%m-%d-%Y").date()
        except Exception:
            return failure_response("Invalid date format", 400)

        # check if poster exists
        if not Person.objects.filter(id=int(poster_id)).exists():
            return failure_response("Person does not exist")
        poster = Person.objects.get(id=int(poster_id))

        # check if entry already exists
        # if it exists, update the entry
        # otherwise, create a new entry
        if not Entry.objects.filter(poster=poster, date=date).exists():
            # Verify all required information is provided
            if entry_description is None or poster_id is None or date is None:
                return failure_response("Missing entry information", 400)
            entry = Entry.objects.create(
                poster=poster,
                date=date,
                entry_description=entry_description,
                emotion=emotion if emotion is not None else top_labels(entry_description)
            )
            return success_response(self._serializer(entry).data, status.HTTP_201_CREATED)
        else:
            entry = Entry.objects.get(poster=poster, date=date)
            
            if entry_description is not None:
                update(entry, "entry_description", entry_description)
                update(entry, "emotion", top_labels(entry_description))
            update(entry, "emotion", emotion)
        entry.save()

        return success_response(self._serializer(entry).data, status.HTTP_201_CREATED)