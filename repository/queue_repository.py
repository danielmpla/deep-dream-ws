from typing import Optional
from typing import List

from entity.queue_item import QueueItem


class QueueRepository:
    @classmethod
    def count_open_jobs(cls) -> int:
        return len(QueueItem.objects(finished=False, canceled=False))

    @classmethod
    def get_next_job(cls) -> Optional[QueueItem]:
        return QueueItem.objects(finished=False, canceled=False).first()

    @classmethod
    def get_job_by_id(cls, id) -> Optional[QueueItem]:
        return QueueItem.objects(id=id).first()

    @classmethod
    def get_open_jobs(cls) -> List[QueueItem]:
        return QueueItem.objects(finished=False, canceled=False)

    @classmethod
    def cancel_job(cls, id):
        job = QueueItem.objects(id=id).first()

        job.canceled = True

        job.save()
