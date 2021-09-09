from typing import Any, List, Optional, Union
import dataclasses


@dataclasses.dataclass
class Payload:
    __slots__ = ('data', 'no_delete')

    data: Any
    no_delete: bool


@dataclasses.dataclass
class Message:
    __slots__ = ('payload', 'rank', 'tag')

    payload: Union[Payload, Any]
    rank: int
    tag: int

    @property
    def data(self):
        payload = self.payload
        return payload.data if isinstance(payload, Payload) else payload


class Conduit:
    def start(self) -> None:
        raise NotImplementedError(self)

    def active(self) -> bool:
        """Return whether the conduit is active."""
        raise NotImplementedError(self)

    def abort(self) -> None:
        """Abort sending and receiving messages, close any worker threads.
        After this point, the conduit may not be used anymore. Used to prevent
        deadlocks when the master thread crashes."""
        raise NotImplementedError(self)

    def wait_and_close(self) -> None:
        raise NotImplementedError(self)

    def data_available(self) -> bool:
        """Return whether there are any available incoming messages."""
        raise NotImplementedError(self)

    def pop_all(self) -> List[Message]:
        """Get and pop all incoming messages."""
        raise NotImplementedError(self)

    def send(self, data: Any, rank: int, tag: int = 5555):
        """Send a message to the given rank."""
        raise NotImplementedError(self)


class SingleRankConduit(Conduit):
    def __init__(self):
        self._running = False
        self.shutdown_by_remote = False
        self.inbox = []
        self.target: Optional['SingleRankConduit'] = None

    def start(self):
        assert not self._running
        self._running = True

    def active(self):
        return self._running

    def abort(self):
        self._running = False

    def wait_and_close(self):
        assert self._running
        self._running = False

    def data_available(self) -> True:
        return bool(self.inbox)

    def pop_all(self):
        messages = self.inbox
        self.inbox = []
        return messages

    def send(self, payload: Union[Payload, Any], rank: int, tag: int = 5555):
        message = Message(payload, rank, tag)
        self.target.inbox.append(message)


def make_single_rank_conduit_pair():
    a = SingleRankConduit()
    b = SingleRankConduit()
    a.target = b
    b.target = a
    return a, b
