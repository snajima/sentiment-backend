import datetime

from api.utils import failure_response
from api.utils import success_response
from django.contrib.auth.models import User

from ..models import Entry
from ..models import Person
from ..apps import emotion_model

class GetDateEntryController:
    def __init__(self, data, request, serializer, id):
        self._request = request
        self._data = data
        self._serializer = serializer
        self._id = id

    def process(self):
        try:
            date = datetime.datetime.strptime(self._data.get("date"), "%m-%d-%Y").date()
        except Exception:
            return failure_response("Invalid date format", 400)
        
        if not Person.objects.filter(id=int(self._id)).exists():
            return failure_response("Person does not exist")
        poster = Person.objects.get(id=self._id)
        entries = Entry.objects.get(poster=poster, date=date)
        return success_response(self._serializer(entries).data, 200)