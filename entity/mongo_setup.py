import mongoengine


def global_init(docker: bool):
    if docker:
        mongoengine.register_connection(alias='core', host='172.17.0.1', port=27017, name='ws', username='root', password='root', authentication_source='admin')
    else:
        mongoengine.register_connection(alias='core', host='localhost', port=27017, name='ws', username='root', password='root', authentication_source='admin')
