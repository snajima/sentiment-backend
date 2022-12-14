from rest_framework import serializers

from .models import Person


class SimplePersonSerializer(serializers.ModelSerializer):
    first_name = serializers.CharField(source="user.first_name")
    last_name = serializers.CharField(source="user.last_name")

    class Meta:
        model = Person
        fields = (
            "id",
            "first_name",
            "last_name",
        )
        read_only_fields = fields