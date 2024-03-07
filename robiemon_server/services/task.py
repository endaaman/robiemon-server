from tinydb import Query, TinyDB
from .db import get_db
from ..lin.config import Config
from ..schemas import Task


class TaskService:
    def __init__(self, config:Config=Depends(), db:TinyDB=Depends(get_db)):
        self.config = config
        self.db = db
        self.table = db.table('tasks')

    def all(self):
        return [Task(id=r.doc_id, **r) for r in self.table.all()]
