from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List

from protocols.common import NetworkConfig, ProtocolResult, generate_messages
from protocols.gbn import GBNSimulator
from protocols.rdt import RDT30Simulator
from protocols.sr import SRSimulator


@dataclass
class Scenario:
    name: str
    config: NetworkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reliable Data Transfer simulator: rdt3.0, Go-Back-N, Selective Repeat"
    )
    parser.add_argument("--protocol", choices=["rdt", "gbn", "sr", "all"], default="all")
    parser.add_argument("--count", type=int, default=12, help="Number of packets/messages to send")
    parser.add_argument("--size", type=int, default=16, help="Payload size per message")
    parser.add_argument("--window", type=int, default=4, help="Window size for GBN and SR")
    parser.add_argument("--timeout", type=int, default=8, help="Fixed timeout (time units)")

    parser.add_argument("--loss", type=float, default=0.0, help="Packet loss probability [0,1]")
    parser.add_argument("--corrupt", type=float, default=0.0, help="Packet corruption probability [0,1]")
    parser.add_argument("--min-delay", type=int, default=1, help="Minimum network delay")
    parser.add_argument("--max-delay", type=int, default=3, help="Maximum network delay")

    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--run-assignment-tests", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    if args.run_assignment_tests:
        return [
            Scenario(
                name="no_loss_no_corruption",
                config=NetworkConfig(loss_prob=0.0, corrupt_prob=0.0, min_delay=1, max_delay=1),
            ),
            Scenario(
                name="packet_loss",
                config=NetworkConfig(loss_prob=0.25, corrupt_prob=0.0, min_delay=1, max_delay=3),
            ),
            Scenario(
                name="packet_corruption",
                config=NetworkConfig(loss_prob=0.0, corrupt_prob=0.25, min_delay=1, max_delay=3),
            ),
            Scenario(
                name="delayed_packets",
                config=NetworkConfig(loss_prob=0.0, corrupt_prob=0.0, min_delay=4, max_delay=12),
            ),
        ]

    return [
        Scenario(
            name="custom",
            config=NetworkConfig(
                loss_prob=args.loss,
                corrupt_prob=args.corrupt,
                min_delay=args.min_delay,
                max_delay=args.max_delay,
            ),
        )
    ]


def print_result(res: ProtocolResult) -> None:
    avg_delay = (
        0.0
        if res.network_stats["scheduled"] == 0
        else res.network_stats["total_delay"] / res.network_stats["scheduled"]
    )

    print(f"Protocol: {res.protocol:7s} | Scenario: {res.scenario}")
    print(
        f"  success={res.success} delivered={len(res.delivered_messages)}/{len(res.original_messages)} "
        f"time={res.total_time} events={res.event_count}"
    )
    print(
        f"  sender_packets={res.sender_packets} retransmissions={res.sender_retransmissions} "
        f"ack_packets={res.ack_packets}"
    )
    print(
        f"  network: dropped={int(res.network_stats['dropped'])} corrupted={int(res.network_stats['corrupted'])} "
        f"scheduled={int(res.network_stats['scheduled'])} avg_delay={avg_delay:.2f}"
    )
    if res.notes:
        for note in res.notes:
            print(f"  note: {note}")
    print("-" * 92)


def main() -> None:
    args = parse_args()

    if args.count < 0:
        raise ValueError("--count must be >= 0")
    if args.size <= 0:
        raise ValueError("--size must be > 0")
    if args.window <= 0:
        raise ValueError("--window must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.min_delay < 0 or args.max_delay < 0 or args.min_delay > args.max_delay:
        raise ValueError("Delay values must satisfy 0 <= min-delay <= max-delay")

    protocols: Dict[str, Callable[..., ProtocolResult]] = {
        "rdt": RDT30Simulator().run,
        "gbn": GBNSimulator().run,
        "sr": SRSimulator().run,
    }

    protocol_order = ["rdt", "gbn", "sr"] if args.protocol == "all" else [args.protocol]
    scenarios = build_scenarios(args)
    messages = generate_messages(count=args.count, payload_size=args.size, seed=args.seed)

    all_results: List[ProtocolResult] = []

    print("\nReliable Data Transfer Simulation")
    print("=" * 92)
    print(
        f"packets={args.count} payload_size={args.size} timeout={args.timeout} "
        f"window={args.window} seed={args.seed}"
    )
    print("=" * 92)

    for scenario in scenarios:
        for i, protocol in enumerate(protocol_order):
            run_seed = args.seed + i + (100 * scenarios.index(scenario))
            if protocol == "rdt":
                result = protocols[protocol](
                    messages=messages,
                    scenario=scenario.name,
                    net_cfg=scenario.config,
                    timeout=args.timeout,
                    seed=run_seed,
                    verbose=args.verbose,
                )
            else:
                result = protocols[protocol](
                    messages=messages,
                    scenario=scenario.name,
                    net_cfg=scenario.config,
                    timeout=args.timeout,
                    window_size=args.window,
                    seed=run_seed,
                    verbose=args.verbose,
                )
            all_results.append(result)
            print_result(result)

    failures = [r for r in all_results if not r.success]
    print(f"Total runs: {len(all_results)}, Passed: {len(all_results) - len(failures)}, Failed: {len(failures)}")


if __name__ == "__main__":
    main()
