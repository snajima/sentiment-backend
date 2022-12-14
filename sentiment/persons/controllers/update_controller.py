from api.utils import success_response
from api.utils import update
from django.contrib.auth.models import User

class UpdatePersonController:
    def __init__(self, user, data, serializer):
        self._data = data
        self._serializer = serializer
        self._user = user
        self._person = self._user.person

    def process(self):
        phone_number = self._data.get("phone_number")
        update(self._person, "phone_number", phone_number)

        self._user.save()
        self._person.save()

        self._user = User.objects.get(id=self._user.id)
        return success_response(self._serializer(self._user).data)