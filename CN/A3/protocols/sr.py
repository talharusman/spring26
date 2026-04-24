from __future__ import annotations

import random
from typing import Dict, List, Set

from .common import EventLoop, NetworkConfig, Packet, ProtocolResult, UnreliableNetwork


class SRSimulator:
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

        snd_base = 0
        snd_next = 0
        snd_packets: Dict[int, Packet] = {}
        snd_acked: Set[int] = set()
        timer_counter = 0
        timer_token_by_seq: Dict[int, int] = {}

        rcv_base = 0
        rcv_buffer: Dict[int, str] = {}
        delivered: List[str] = []

        data_packets_sent = 0
        ack_packets_sent = 0
        retransmissions = 0
        event_count = 0

        def log(msg: str) -> None:
            if verbose:
                print(msg)

        def start_timer(seq: int) -> None:
            nonlocal timer_counter
            timer_counter += 1
            token = timer_counter
            timer_token_by_seq[seq] = token
            loop.schedule(
                delay=timeout,
                event_type="timeout",
                payload={"seq": seq, "token": token},
            )

        def stop_timer(seq: int) -> None:
            timer_token_by_seq.pop(seq, None)

        def send_while_window_open() -> None:
            nonlocal snd_next, data_packets_sent
            while snd_next < total and snd_next < snd_base + window_size:
                pkt = Packet.make_data(seq_num=snd_next, payload=messages[snd_next])
                snd_packets[snd_next] = pkt
                data_packets_sent += 1
                log(f"[SR] SENDER send seq={snd_next}")
                network.send(packet=pkt, event_loop=loop, event_type="to_receiver")
                start_timer(snd_next)
                snd_next += 1

        send_while_window_open()

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
                    continue

                seq = pkt.seq_num
                if seq < rcv_base:
                    # Duplicate data packet outside receiver window (already delivered).
                    ack = Packet.make_ack(seq)
                    ack_packets_sent += 1
                    network.send(packet=ack, event_loop=loop, event_type="to_sender")
                    continue

                if seq >= rcv_base + window_size:
                    # Packet is ahead of receiver window; ignore per SR rules.
                    continue

                if seq not in rcv_buffer:
                    rcv_buffer[seq] = pkt.payload
                    log(f"[SR] RECEIVER buffer seq={seq}")

                ack = Packet.make_ack(seq)
                ack_packets_sent += 1
                network.send(packet=ack, event_loop=loop, event_type="to_sender")

                while rcv_base in rcv_buffer:
                    delivered.append(rcv_buffer.pop(rcv_base))
                    rcv_base += 1

            elif event_type == "to_sender":
                ack_pkt: Packet = payload["packet"]
                if ack_pkt.is_corrupt() or not ack_pkt.is_ack or ack_pkt.ack_num is None:
                    continue

                ack_seq = ack_pkt.ack_num
                if ack_seq in snd_packets and ack_seq not in snd_acked:
                    snd_acked.add(ack_seq)
                    stop_timer(ack_seq)
                    log(f"[SR] SENDER got ACK seq={ack_seq}")

                    while snd_base in snd_acked:
                        snd_acked.remove(snd_base)
                        snd_packets.pop(snd_base, None)
                        snd_base += 1

                    send_while_window_open()

            elif event_type == "timeout":
                seq = payload["seq"]
                token = payload["token"]
                if timer_token_by_seq.get(seq) != token:
                    continue
                if seq not in snd_packets:
                    continue

                pkt = snd_packets[seq]
                data_packets_sent += 1
                retransmissions += 1
                log(f"[SR] TIMEOUT seq={seq}, retransmit")
                network.send(packet=pkt, event_loop=loop, event_type="to_receiver")
                start_timer(seq)

            done = snd_base >= total and len(delivered) == total
            if done:
                break

        success = delivered == messages
        notes = []
        if event_count >= max_events:
            notes.append("Reached max event budget; potential infinite loop.")
        if not success:
            notes.append("Delivered payload differs from original payload list.")

        return ProtocolResult(
            protocol="sr",
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
