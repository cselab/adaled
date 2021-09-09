from adaled.conduits.base import Conduit, Message, Payload
from adaled.utils.threading_ import ThreadEx

from mpi4py import MPI

from typing import Any, List, Optional, Tuple, Union
import os
import threading
import time
import sys


class IntercommConduit(Conduit):
    """One-to-one communication over an intercommunicator.

    Each rank can send arbitrary messages to any rank on the remote side.

    To close the connection, the user must invoke `wait_and_close()` on both
    groups, collectively within each group. The Boolean attribute
    `shutdown_by_remote` can be used to check if the remote side already
    invoked `wait_and_close()`.

    Any messages sent before invoking `wait_and_close()` will be received by
    the remote.
    """
    TAG_SHUTDOWN = 5554

    def __init__(self,
                 intercomm: MPI.Intercomm,
                 max_messages: int = sys.maxsize,
                 debug: Optional[bool] = None):
        """
        Arguments:
            debug: (bool, optional) If set to `True`, print debug information.
                If not specified or equal to `None`, the environment variable
                `ADALED_DEBUG_CONDUITS` will be considered instead.
                Note that in the debug mode, a larger sleep time used.
        """
        assert isinstance(max_messages, int), max_messages
        self.intercomm = intercomm
        self.shutdown_by_remote = False
        self.max_messages = max_messages
        self.debug = debug if debug is not None \
                else int(os.environ.get('ADALED_DEBUG_CONDUITS') or '0')

        self._stopping = False
        self._abort_event = threading.Event()

        self._lock = threading.Lock()
        self._thread = None
        self._inbox: List[Message] = []
        self._outbox: List[Message] = []

    def log(self, *args, flush=True, **kwargs):
        if self.debug:
            print(*args, flush=flush, **kwargs)

    def active(self):
        return self._thread is not None

    def start(self):
        assert not self.active()
        self._abort_event.clear()
        self._thread = ThreadEx(target=self._run)
        self._thread.start()
        self.shutdown_by_remote = False
        self._stopping = False

    def abort(self):
        self.log("ABORT")
        if self._thread:
            self._abort_event.set()
            self._thread.join()

    def wait_and_close(self):
        """Send a close connection message to the remote group, wait for all
        request to finish and stop listening for the requests."""
        self.log("WAIT AND CLOSE")
        assert self.active()
        self._send_shutdown()
        self._stopping = True
        t0 = time.time()
        self._thread.join()
        self._thread = None
        dt = time.time() - t0
        self.log(f"JOIN COMPLETED IN {dt:.6f}s")

    def data_available(self):
        with self._lock:
            return bool(self._inbox)

    def pop_all(self) -> List[Message]:
        with self._lock:
            out = self._inbox
            self._inbox = []
        return out

    def send(self, payload: Any, target_rank: int, tag: int = 5555):
        if tag == self.TAG_SHUTDOWN:
            raise ValueError(f"tag {tag} is reserved")
        if self._stopping:
            raise RuntimeError(
                    "already stopping or stopped, cannot send more messages")
        self._thread.check_exception()
        with self._lock:
            self._outbox.append(Message(payload, target_rank, tag))

    def _send_shutdown(self):
        """Send shutdown signals to the remote group. Collective operation."""
        with self._lock:
            inter = self.intercomm
            for target_rank in range(inter.rank, inter.remote_size, inter.size):
                self._outbox.append(Message(0, target_rank, self.TAG_SHUTDOWN))

    def _run(self):
        # Stopping mechanism based on the NBX algorithm from
        #   T. Hoefler, C. Siebert, A. Lumsdaine (2010),
        #   "Scalable Communication Protocols for Dynamic Sparse Data Exchange"
        incoming: List[MPI.Request] = []
        outgoing: List[MPI.Request] = []
        status = MPI.Status()
        self.log("RUN")

        ibarrier = None
        while not self._abort_event.is_set():
            # Any new incoming messages?
            while True:
                message = self.intercomm.improbe(status=status)
                if message:
                    self.log(f"INCOMING src={status.source} tag={status.tag}")
                    incoming.append(message.irecv())
                    if status.tag == self.TAG_SHUTDOWN:
                        self.shutdown_by_remote = True
                else:
                    break

            # Check the status of known incoming messages.
            incoming = self._process_incoming_requests(incoming)

            # Anything to send?
            with self._lock:
                to_send: List[Message] = self._outbox
                self._outbox = []

            # Check the status of outgoing requests.
            outgoing = [r for r in outgoing if not r.Test()]
            for message in to_send:
                self.log("SEND", message.data.__class__, message.rank, message.tag)
                # Using issend in order to have information of whether the
                # target started receiving the message.
                outgoing.append(self.intercomm.issend(
                        message.payload, message.rank, message.tag))

            if self._stopping and not outgoing:
                # The ibarrier means that all our outgoing requests had a
                # matching remote irecv. However, we do not know if there are
                # any irecv that we have to still initiate. Thus, we continue
                # the loop until the ibarrier completes.
                # Note #1: this ibarrier only synchronizes one group with
                # another and not ranks within the same group. Since we do not
                # exchange anything within the group itself, this is fine.
                # Note #2: mixing Ibarrier and send/recv is allowed, see
                # example 5.33 (page 220) here:
                # https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf
                if not ibarrier:
                    self.log("IBARRIER START")
                    ibarrier = self.intercomm.Ibarrier()
                elif ibarrier is not True:
                    if ibarrier.Test():
                        self.log("IBARRIER SUCCEEDED")
                        ibarrier = True
                elif not incoming:
                    self.log("ALL REQUESTS DONE")
                    break  # All messages received, stop.

            self.log("SLEEP, INCOMING:", len(incoming),
                     " OUTGOING:", len(outgoing),
                     " STOPPING:", self._stopping, ibarrier)
            time.sleep(0.1 if self.debug else 0.01)  # Requests are really rare.

    def _process_incoming_requests(self, incoming: List[MPI.Request]) \
            -> List[MPI.Request]:
        """Put finished requests into the inbox and return the list of
        unfinished requests."""
        status = MPI.Status()
        unfinished = []
        for request in incoming:
            completed, data = request.test(status)
            if completed:
                if status.tag == self.TAG_SHUTDOWN:
                    self.log("MESSAGE SHUTDOWN RECEIVED", status.source)
                    continue
                with self._lock:
                    self._free_inbox_space()
                    msg = Message(data, status.source, status.tag)
                    self._inbox.append(msg)
                    self.log("MESSAGE RECEIVED",
                             msg.data.__class__, msg.rank, msg.tag)
            else:
                unfinished.append(request)
        return unfinished

    def _free_inbox_space(self):
        """Remove a message if the inbox is about to get full. Keep `no_delete`
        messages. Internal function, must be invoked under a lock."""
        if len(self._inbox) < self.max_messages:
            return
        for i in range(len(self._inbox)):
            msg = self._inbox[i]
            payload = msg.payload
            if not (isinstance(payload, Payload) and payload.no_delete):
                self.log("DROPPING MESSAGE", msg.data.__class__, msg.rank, msg.tag)
                del self._inbox[i]
                break
