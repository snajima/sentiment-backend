import json

from api.utils import success_response
from django.contrib.auth.models import User
from rest_framework import generics
from rest_framework import status

from .controllers.authenticate_controller import AuthenticateController
from .controllers.developer_controller import DeveloperController
from .controllers.update_controller import UpdatePersonController
from .serializers import AuthenticateSerializer
from .serializers import UserSerializer


class AuthenticateView(generics.GenericAPIView):
    serializer_class = AuthenticateSerializer

    def post(self, request):
        """Authenticate the current user."""
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = request.data
        return AuthenticateController(request, data, self.serializer_class).process()


class MeView(generics.GenericAPIView):
    serializer_class = UserSerializer

    def get(self, request):
        """Get current authenticated user."""
        return success_response(
            self.serializer_class(request.user).data, status.HTTP_200_OK
        )

    def post(self, request):
        """Update current authenticated user."""
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            data = request.data
        return UpdatePersonController(
            request.user, data, self.serializer_class
        ).process()

class DeveloperView(generics.GenericAPIView):
    serializer_class = AuthenticateSerializer

    def get(self, request):
        """Get all users."""
        users = [UserSerializer(user).data for user in User.objects.all()]
        return success_response({"users": users})

    def post(self, request):
        """Create test user or return access token for test user if `id` is provided."""
        try:
            data = json.loads(request.body)
            id = request.query_params.get("id", None)
        except json.JSONDecodeError:
            data = request.data
        return DeveloperController(request, data, self.serializer_class, id).process()