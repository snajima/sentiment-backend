from entries.models import Entry
from rest_framework import serializers

class EntrySerializer(serializers.ModelSerializer):
    class Meta:
        model = Entry
        fields = (
            "id",
            "entry_description",
            # "first_name",
            # "last_name",
            "emotion",
        )
        read_only_fields = fields