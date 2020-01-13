from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from flask import request

from io import BytesIO

from flask_cors import CORS

from entity import mongo_setup
from entity.queue_item import QueueItem
from repository.queue_repository import QueueRepository
from test import DeepDreamWS

app = Flask(__name__)
CORS(app)


job_defaults = {
    'coalesce': False,
    'max_instances': 1
}

scheduler = BackgroundScheduler(job_defaults=job_defaults)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'Image missing', 404, {'Content-Type': 'text/plain'}

    file = request.files['image']
    file_content = BytesIO()

    file.save(file_content)

    job = QueueItem()

    job.base_image = file_content.getvalue()

    file_content.close()

    job = job.save()

    return str(job.id), 200, {'Content-Type': 'text/plain'}


@app.route('/image/<id>', methods=['GET'])
def get_image(id):
    job = QueueRepository.get_job_by_id(id)

    if not job.finished:
        return str(job.id), 202, {'Content-Type': 'text/plain'}

    image = job.computed_image

    return image, 200, {'Content-Type': 'image/jpeg'}


def job_runner():

    jobs = QueueRepository.get_open_jobs()

    for i in range(len(jobs)):
        id = str(jobs[i].id)

        dd = DeepDreamWS()

        dd.run(id)


if __name__ == '__main__':
    mongo_setup.global_init(False)
    scheduler.add_job(job_runner, 'interval', minutes=1, id='deep-dream-job')
    scheduler.start()
    app.run(host='0.0.0.0', port=5000)
    scheduler.shutdown()
