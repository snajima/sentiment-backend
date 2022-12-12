from random import randint

from api.utils import failure_response
from api.utils import success_response
from django.contrib.auth.models import User
from persons.models import Person
from rest_framework import status
from rest_framework.authtoken.models import Token


class DeveloperController:
    def __init__(self, request, data, serializer, id):
        self._request = request
        self._data = data
        self._serializer = serializer
        self._user_id = id

    def create_person(self, person_data):
        """Creates new Person object from `person_data`."""
        person = Person(**person_data)
        person.save()
        return person

    def create_user(self, user_data):
        """Creates new user (Django auth) from `user_data`."""
        return User.objects._create_user(**user_data)

    def process(self):
        """Creates new user and person if user_id not provided in body. Otherwise, return access token for valid user."""
        status_code = status.HTTP_200_OK
        if self._user_id is None:
            user_data = {
                "username": self._data.get("username"),
                "email": self._data.get("email"),
                "password": self._data.get("password"),
                "first_name": self._data.get("first_name"),
                "last_name": self._data.get("last_name"),
            }
            user = self.create_user(user_data)
            person_data = {"user": user}
            self.create_person(person_data)
            status_code = status.HTTP_201_CREATED
        else:
            if not User.objects.filter(id=int(self._user_id)).exists():
                return failure_response("User does not exist")
            user = User.objects.get(id=int(self._user_id))
        access_token, _ = Token.objects.get_or_create(user=user)
        return success_response(
            self._serializer(user, context={"access_token": access_token.key}).data,
            status_code,
        )