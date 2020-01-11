import datetime
import uuid

import mongoengine


class QueueItem(mongoengine.Document):
    # id = mongoengine.StringField(default=str(uuid.uuid1()), primary_key=True, unique=True)
    created = mongoengine.DateTimeField(default=datetime.datetime.now)

    base_image = mongoengine.BinaryField()
    computed_image = mongoengine.BinaryField()

    finished = mongoengine.BooleanField(default=False)
    canceled = mongoengine.BooleanField(default=False)

    meta = {
        'db_alias': 'core',
        'collection': 'queueItem',
        'indexes': [
            'created'
        ],
        'ordering': ['-created']
    }
