from flask import Flask
from flask import request

from io import BytesIO

from flask_cors import CORS

from entity import mongo_setup
from entity.queue_item import QueueItem
from repository.queue_repository import QueueRepository

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        print(request.files)

    file = request.files['image']
    file_content = BytesIO()

    file.save(file_content)

    job = QueueItem()

    job.base_image = file_content.getvalue()

    file_content.close()

    job = job.save()

    return str(job.id)


@app.route('/image/<id>', methods=['GET'])
def get_image(id):
    job = QueueRepository.get_job_by_id(id)

    image = job.base_image

    return image, 200, {'Content-Type': 'image/jpeg'}


if __name__ == '__main__':
    mongo_setup.global_init()
    app.run(host='0.0.0.0', port=5000, debug=True)
