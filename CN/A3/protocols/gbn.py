from __future__ import annotations

import random
from typing import Dict, List

from .common import EventLoop, NetworkConfig, Packet, ProtocolResult, UnreliableNetwork


class GBNSimulator:
    def run(
        self,
        messages: List[str],
        scenario: str,
        net_cfg: NetworkConfig,
        timeout: int,
        window_size: int,
        seed: int,
        verbose: bool = False,
    ) -> ProtocolResult:
        rng = random.Random(seed)
        network = UnreliableNetwork(cfg=net_cfg, rng=rng)
        loop = EventLoop()

        total = len(messages)
        base = 0
        next_seq = 0
        sender_packets: Dict[int, Packet] = {}

        timer_token = 0
        active_timer_token = 0

        receiver_expected = 0
        last_in_order_ack = -1
        delivered: List[str] = []

        data_packets_sent = 0
        ack_packets_sent = 0
        retransmissions = 0
        event_count = 0

        def log(msg: str) -> None:
            if verbose:
                print(msg)

        def start_timer() -> None:
            nonlocal timer_token, active_timer_token
            if base >= next_seq:
                active_timer_token = 0
                return
            timer_token += 1
            active_timer_token = timer_token
            loop.schedule(
                delay=timeout,
                event_type="timeout",
                payload={"token": active_timer_token, "base": base},
            )

        def send_new_packets() -> None:
            nonlocal next_seq, data_packets_sent
            while next_seq < total and next_seq < base + window_size:
                start_base_edge = base == next_seq
                pkt = Packet.make_data(seq_num=next_seq, payload=messages[next_seq])
                sender_packets[next_seq] = pkt
                data_packets_sent += 1
                log(f"[GBN] SENDER send seq={next_seq}")
                network.send(packet=pkt, event_loop=loop, event_type="to_receiver")
                next_seq += 1
                if start_base_edge:
                    start_timer()

        send_new_packets()

        max_events = 1_000_000
        while loop.has_events() and event_count < max_events:
            popped = loop.pop()
            if popped is None:
                break
            _, event_type, payload = popped
            event_count += 1

            if event_type == "to_receiver":
                pkt: Packet = payload["packet"]
                if pkt.is_corrupt() or pkt.is_ack:
                    ack = Packet.make_ack(last_in_order_ack)
                    ack_packets_sent += 1
                    network.send(packet=ack, event_loop=loop, event_type="to_sender")
                    continue

                if pkt.seq_num == receiver_expected:
                    delivered.append(pkt.payload)
                    last_in_order_ack = receiver_expected
                    receiver_expected += 1
                    log(f"[GBN] RECEIVER accept seq={pkt.seq_num}")
                else:
                    log(
                        f"[GBN] RECEIVER discard seq={pkt.seq_num}, expected={receiver_expected}"
                    )

                ack = Packet.make_ack(last_in_order_ack)
                ack_packets_sent += 1
                network.send(packet=ack, event_loop=loop, event_type="to_sender")

            elif event_type == "to_sender":
                ack_pkt: Packet = payload["packet"]
                if ack_pkt.is_corrupt() or not ack_pkt.is_ack or ack_pkt.ack_num is None:
                    continue

                ack_num = ack_pkt.ack_num
                if ack_num >= base and ack_num < next_seq:
                    base = ack_num + 1
                    log(f"[GBN] SENDER cumulative ACK={ack_num}, new base={base}")
                    if base == next_seq:
                        active_timer_token = 0
                    else:
                        start_timer()
                    send_new_packets()

            elif event_type == "timeout":
                if payload["token"] != active_timer_token:
                    continue
                if payload["base"] != base:
                    continue
                if base >= next_seq:
                    continue

                log(f"[GBN] TIMEOUT at base={base}, retransmit window [{base}, {next_seq - 1}]")
                for seq in range(base, next_seq):
                    pkt = sender_packets[seq]
                    data_packets_sent += 1
                    retransmissions += 1
                    network.send(packet=pkt, event_loop=loop, event_type="to_receiver")
                start_timer()

            done = base >= total and len(delivered) == total
            if done:
                break

        success = delivered == messages
        notes = []
        if event_count >= max_events:
            notes.append("Reached max event budget; potential infinite loop.")
        if not success:
            notes.append("Delivered payload differs from original payload list.")

        return ProtocolResult(
            protocol="gbn",
            scenario=scenario,
            success=success,
            original_messages=messages,
            delivered_messages=delivered,
            total_time=loop.current_time,
            sender_packets=data_packets_sent,
            ack_packets=ack_packets_sent,
            sender_retransmissions=retransmissions,
            event_count=event_count,
            network_stats=network.stats,
            notes=notes,
        )
