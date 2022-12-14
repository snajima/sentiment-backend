from django.contrib.auth.models import User
from rest_framework import serializers

class AuthenticateSerializer(serializers.ModelSerializer):
    access_token = serializers.SerializerMethodField(method_name="get_access_token")

    class Meta:
        model = User
        fields = (
            "access_token",
            "username",
            "first_name",
            "last_name",
        )
        read_only_fields = fields

    def get_access_token(self, instance):
        return self.context.get("access_token")


class UserSerializer(serializers.ModelSerializer):
    profile_pic_url = serializers.CharField(source="person.profile_pic_url")
    phone_number = serializers.CharField(source="person.phone_number")

    class Meta:
        model = User
        fields = (
            "id",
            "first_name",
            "last_name",
            "phone_number",
            "profile_pic_url",
        )
        read_only_fields = fields