from __future__ import annotations

import random
from typing import List

from .common import EventLoop, NetworkConfig, Packet, ProtocolResult, UnreliableNetwork


class RDT30Simulator:
    def run(
        self,
        messages: List[str],
        scenario: str,
        net_cfg: NetworkConfig,
        timeout: int,
        seed: int,
        verbose: bool = False,
    ) -> ProtocolResult:
        rng = random.Random(seed)
        network = UnreliableNetwork(cfg=net_cfg, rng=rng)
        loop = EventLoop()

        sender_index = 0
        sender_seq = 0
        waiting_for_ack = False
        in_flight_packet: Packet | None = None
        timer_token = 0
        active_timer_token = 0

        receiver_expected_seq = 0
        receiver_last_ack = -1
        delivered: List[str] = []

        data_packets_sent = 0
        ack_packets_sent = 0
        retransmissions = 0
        event_count = 0

        def log(msg: str) -> None:
            if verbose:
                print(msg)

        def start_timer(index: int, seq_num: int) -> None:
            nonlocal timer_token, active_timer_token
            timer_token += 1
            active_timer_token = timer_token
            loop.schedule(
                delay=timeout,
                event_type="timeout",
                payload={"index": index, "seq_num": seq_num, "token": active_timer_token},
            )

        def send_current_packet(index: int) -> None:
            nonlocal waiting_for_ack, in_flight_packet, data_packets_sent
            payload = messages[index]
            packet = Packet.make_data(seq_num=sender_seq, payload=payload)
            in_flight_packet = packet
            waiting_for_ack = True
            data_packets_sent += 1
            log(f"[RDT] SENDER send seq={packet.seq_num} index={index}")
            network.send(packet=packet, event_loop=loop, event_type="to_receiver")
            start_timer(index=index, seq_num=sender_seq)

        if messages:
            send_current_packet(index=0)

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
                    ack = Packet.make_ack(receiver_last_ack)
                    ack_packets_sent += 1
                    network.send(packet=ack, event_loop=loop, event_type="to_sender")
                    continue

                if pkt.seq_num == receiver_expected_seq:
                    delivered.append(pkt.payload)
                    receiver_last_ack = pkt.seq_num
                    receiver_expected_seq += 1
                    log(f"[RDT] RECEIVER accept seq={pkt.seq_num}")
                else:
                    log(f"[RDT] RECEIVER duplicate seq={pkt.seq_num}, expected={receiver_expected_seq}")

                ack = Packet.make_ack(receiver_last_ack)
                ack_packets_sent += 1
                network.send(packet=ack, event_loop=loop, event_type="to_sender")

            elif event_type == "to_sender":
                ack_pkt: Packet = payload["packet"]
                if ack_pkt.is_corrupt() or not ack_pkt.is_ack:
                    continue

                if waiting_for_ack and ack_pkt.ack_num == sender_seq:
                    waiting_for_ack = False
                    sender_index += 1
                    sender_seq += 1
                    log(f"[RDT] SENDER got ACK={ack_pkt.ack_num}, advance to index={sender_index}")
                    if sender_index < len(messages):
                        send_current_packet(index=sender_index)

            elif event_type == "timeout":
                if not waiting_for_ack:
                    continue
                if payload["token"] != active_timer_token:
                    continue
                if payload["seq_num"] != sender_seq or payload["index"] != sender_index:
                    continue

                if in_flight_packet is None:
                    continue

                retransmissions += 1
                data_packets_sent += 1
                log(f"[RDT] TIMEOUT seq={sender_seq}, retransmit")
                network.send(packet=in_flight_packet, event_loop=loop, event_type="to_receiver")
                start_timer(index=sender_index, seq_num=sender_seq)

            done = sender_index >= len(messages) and not waiting_for_ack and len(delivered) == len(messages)
            if done:
                break

        success = delivered == messages
        notes = []
        if event_count >= max_events:
            notes.append("Reached max event budget; potential infinite loop.")
        if not success:
            notes.append("Delivered payload differs from original payload list.")

        return ProtocolResult(
            protocol="rdt3.0",
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
