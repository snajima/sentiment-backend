# Generated by Django 4.1.2 on 2022-12-03 05:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('persons', '0002_remove_person_netid'),
    ]

    operations = [
        migrations.AddField(
            model_name='person',
            name='netid',
            field=models.CharField(default='', max_length=10, unique=True),
            preserve_default=False,
        ),
    ]
