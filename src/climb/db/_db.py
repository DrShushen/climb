import abc
from typing import List

from climb.common import Session, UserSettings


# TODO: Some unique identifier for the user.
class DB(abc.ABC):
    @abc.abstractmethod
    def update_user_settings(self, settings: UserSettings) -> None: ...

    @abc.abstractmethod
    def get_user_settings(self) -> UserSettings: ...

    @abc.abstractmethod
    def update_session(self, session: Session) -> None: ...

    @abc.abstractmethod
    def get_session(self, session_key: str) -> Session: ...

    @abc.abstractmethod
    def get_all_sessions(self) -> List[Session]: ...

    @abc.abstractmethod
    def delete_session(self, session_key: str) -> None: ...
