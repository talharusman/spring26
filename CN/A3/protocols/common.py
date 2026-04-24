from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import random
import string
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Packet:
    seq_num: int
    ack_num: Optional[int]
    payload: str
    is_ack: bool
    checksum: int

    @staticmethod
    def _checksum(seq_num: int, ack_num: Optional[int], payload: str, is_ack: bool) -> int:
        payload_sum = sum(ord(ch) for ch in payload)
        ack_value = ack_num if ack_num is not None else 0
        return seq_num * 31 + ack_value * 17 + payload_sum + (97 if is_ack else 53)

    @classmethod
    def make_data(cls, seq_num: int, payload: str) -> "Packet":
        checksum = cls._checksum(seq_num=seq_num, ack_num=None, payload=payload, is_ack=False)
        return cls(seq_num=seq_num, ack_num=None, payload=payload, is_ack=False, checksum=checksum)

    @classmethod
    def make_ack(cls, ack_num: int) -> "Packet":
        checksum = cls._checksum(seq_num=0, ack_num=ack_num, payload="", is_ack=True)
        return cls(seq_num=0, ack_num=ack_num, payload="", is_ack=True, checksum=checksum)

    def is_corrupt(self) -> bool:
        expected = self._checksum(
            seq_num=self.seq_num,
            ack_num=self.ack_num,
            payload=self.payload,
            is_ack=self.is_ack,
        )
        return expected != self.checksum

    def as_corrupted_copy(self) -> "Packet":
        return Packet(
            seq_num=self.seq_num,
            ack_num=self.ack_num,
            payload=self.payload,
            is_ack=self.is_ack,
            checksum=self.checksum + 1,
        )


@dataclass
class NetworkConfig:
    loss_prob: float = 0.0
    corrupt_prob: float = 0.0
    min_delay: int = 1
    max_delay: int = 1


@dataclass
class ProtocolResult:
    protocol: str
    scenario: str
    success: bool
    original_messages: List[str]
    delivered_messages: List[str]
    total_time: int
    sender_packets: int
    ack_packets: int
    sender_retransmissions: int
    event_count: int
    network_stats: Dict[str, float]
    notes: List[str] = field(default_factory=list)


class EventLoop:
    def __init__(self) -> None:
        self.current_time = 0
        self._counter = 0
        self._pq: List[Tuple[int, int, str, Dict[str, Any]]] = []

    def schedule(self, delay: int, event_type: str, payload: Dict[str, Any]) -> None:
        self._counter += 1
        event_time = self.current_time + max(0, delay)
        heapq.heappush(self._pq, (event_time, self._counter, event_type, payload))

    def pop(self) -> Optional[Tuple[int, str, Dict[str, Any]]]:
        if not self._pq:
            return None
        event_time, _, event_type, payload = heapq.heappop(self._pq)
        self.current_time = event_time
        return event_time, event_type, payload

    def has_events(self) -> bool:
        return bool(self._pq)


class UnreliableNetwork:
    def __init__(self, cfg: NetworkConfig, rng: random.Random) -> None:
        self.cfg = cfg
        self.rng = rng
        self.stats: Dict[str, float] = {
            "scheduled": 0,
            "dropped": 0,
            "corrupted": 0,
            "total_delay": 0,
            "data_scheduled": 0,
            "ack_scheduled": 0,
        }

    def send(
        self,
        packet: Packet,
        event_loop: EventLoop,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = payload or {}

        if self.rng.random() < self.cfg.loss_prob:
            self.stats["dropped"] += 1
            return

        out_packet = packet
        if self.rng.random() < self.cfg.corrupt_prob:
            out_packet = packet.as_corrupted_copy()
            self.stats["corrupted"] += 1

        delay = self.rng.randint(self.cfg.min_delay, self.cfg.max_delay)
        self.stats["scheduled"] += 1
        self.stats["total_delay"] += delay
        if packet.is_ack:
            self.stats["ack_scheduled"] += 1
        else:
            self.stats["data_scheduled"] += 1

        event_payload = dict(payload)
        event_payload["packet"] = out_packet
        event_loop.schedule(delay=delay, event_type=event_type, payload=event_payload)


def generate_messages(count: int, payload_size: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    alphabet = string.ascii_uppercase + string.digits
    messages = []
    for idx in range(count):
        body = "".join(rng.choice(alphabet) for _ in range(max(1, payload_size - 6)))
        messages.append(f"MSG{idx:03d}-{body}")
    return messages
