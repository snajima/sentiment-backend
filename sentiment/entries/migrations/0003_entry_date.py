# Generated by Django 4.1.2 on 2022-12-03 05:29

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('entries', '0002_remove_entry_user_entry_poster_alter_entry_emotion_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='entry',
            name='date',
            field=models.DateTimeField(default=datetime.datetime(2022, 12, 3, 5, 29, 22, 453160)),
            preserve_default=False,
        ),
    ]
