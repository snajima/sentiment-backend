from entries.models import Entry
from rest_framework import serializers
from persons.simple_serializers import SimplePersonSerializer

class EntrySerializer(serializers.ModelSerializer):
    poster = SimplePersonSerializer()

    class Meta:
        model = Entry
        fields = (
            "id",
            "poster",
            "entry_description",
            "emotion",
            "date"
        )
        read_only_fields = fields