from api.utils import failure_response
from api.utils import success_response

from ..models import Entry
from ..models import Person

class GetPersonEntryController:
    def __init__(self, request, serializer, id):
        self._request = request
        self._serializer = serializer
        self._id = id

    def process(self):
        if not Person.objects.filter(id=int(self._id)).exists():
            return failure_response("Person does not exist")
        poster = Person.objects.get(id=self._id)

        entries = Entry.objects.filter(poster=poster)
        # ({"entries": self.serializer_class(self.queryset.all(), many=True).data})
        return success_response({"entries": self._serializer(entries, many=True).data}, 200)