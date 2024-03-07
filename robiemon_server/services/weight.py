from tinydb import Query, TinyDB
from fastapi import Depends
from .db import get_db
from ..lin.config import Config
from ..schemas import Weight


class BTResultService:
    def __init__(self, config:Config=Depends(), db:TinyDB=Depends(get_db)):
        self.config = config
        self.db = db
        self.table = db.table('weights')

    def all(self):
        return [Weight(id=r.doc_id, **r) for r in self.table.all()]
