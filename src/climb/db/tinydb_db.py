import enum
from typing import List

from tinydb import Query, TinyDB
from tinydb.storages import JSONStorage
from tinydb.table import Document
from tinydb_serialization import SerializationMiddleware, Serializer
from tinydb_serialization.serializers import DateTimeSerializer

from climb.common import Session, UserSettings
from climb.common.serialization import (
    decode_enum,
    encode_enum,
    session_from_serializable_dict,
    session_to_serializable_dict,
)

from ._db import DB


# Custom serializer for enums.
class EnumSerializer(Serializer):
    OBJ_CLASS = enum.Enum  # The class this serializer handles

    def encode(self, obj: enum.Enum) -> str:
        return encode_enum(obj)

    def decode(self, s: str) -> enum.Enum:
        return decode_enum(s)


serialization = SerializationMiddleware(JSONStorage)
serialization.register_serializer(DateTimeSerializer(), "TinyDate")
serialization.register_serializer(EnumSerializer(), "TinyEnum")


class TinyDB_DB(DB):
    def __init__(self, db_path: str = "db.json") -> None:
        self.db_path = db_path
        self.db = TinyDB(db_path, storage=serialization)

    def update_user_settings(self, settings: UserSettings) -> None:
        # This "table" is used only to store user settings, use Document/doc_id upsert method to update this.
        self.db.table("user_settings").upsert(Document(settings.model_dump(), doc_id=0))

    def get_user_settings(self) -> UserSettings:
        exists = len(self.db.table("user_settings").all()) > 0
        if not exists:
            self.update_user_settings(UserSettings())
        # Retrieve the first (and only) user settings document:
        return UserSettings(**self.db.table("user_settings").all()[0])

    def update_session(self, session: Session) -> None:
        serializable_session = session_to_serializable_dict(session)
        self.db.table("session").upsert(serializable_session, Query().session_key == session.session_key)

    def get_session(self, session_key: str) -> Session:
        doc_to_deserialize = self.db.table("session").search(Query().session_key == session_key)[0]
        deserialized_session = session_from_serializable_dict(doc_to_deserialize)
        return deserialized_session

    def get_all_sessions(self) -> List[Session]:
        return [session_from_serializable_dict(doc) for doc in self.db.table("session").all()]

    def delete_session(self, session_key: str) -> None:
        self.db.table("session").remove(Query().session_key == session_key)
