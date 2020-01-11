import mongoengine


def global_init():
    mongoengine.register_connection(alias='core', host='localhost', port=27017, name='ws', username='root', password='root', authentication_source='admin')
