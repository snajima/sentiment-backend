import datetime

from api.utils import failure_response
from api.utils import success_response

from ..models import Entry
from ..models import Person

class GetRangeEntryController:
    def __init__(self, data, request, serializer, id):
        self._request = request
        self._data = data
        self._serializer = serializer
        self._id = id

    def process(self):
        try:
            start_date = datetime.datetime.strptime(self._data.get("start_date"), "%m-%d-%Y").date()
            end_date = datetime.datetime.strptime(self._data.get("end_date"), "%m-%d-%Y").date()
        except Exception:
            return failure_response("Invalid date format", 400)
        
        if not Person.objects.filter(id=int(self._id)).exists():
            return failure_response("Person does not exist")
        poster = Person.objects.get(id=self._id)

        # get all entries between start and end date
        if not Entry.objects.filter(poster=poster, date__range=[start_date, end_date]).exists():
            return failure_response("Entry does not exist")
        entries = Entry.objects.filter(poster=poster, date__range=[start_date, end_date])

        # convert list of entries to dictionary with emotion as key and percentage as value
        emotion_distributions = {}
        for entry in entries:
            emotion_distributions[entry.emotion] = emotion_distributions.get(entry.emotion, 0) + 1
        for emotion in emotion_distributions:
            emotion_distributions[emotion] = emotion_distributions[emotion] / len(entries)

        return success_response(dict(sorted(emotion_distributions.items(), key=lambda item: item[1], reverse=True)), 200)