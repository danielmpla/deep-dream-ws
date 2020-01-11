from typing import Optional

from entity.queue_item import QueueItem


class QueueRepository:
    @classmethod
    def count_open_jobs(cls):
        pass

    @classmethod
    def get_next_job(cls):
        pass

    @classmethod
    def get_job_by_id(cls, id) -> Optional[QueueItem]:
        return QueueItem.objects(id=id).first()

    @classmethod
    def get_open_jobs(cls):
        pass

    @classmethod
    def cancel_job(cls, uuid):
        pass
