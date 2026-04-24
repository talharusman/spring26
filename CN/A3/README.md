# Reliable Data Transfer Simulation (rdt 3.0, GBN, SR)

This project simulates reliable uni-directional data transfer over an unreliable channel.

Implemented protocols:

- rdt 3.0 (Stop-and-Wait)
- Go-Back-N (GBN)
- Selective Repeat (SR)

The simulator introduces random packet loss, packet corruption, and variable delay, and verifies that all messages are eventually delivered in order.

## Requirements

- Python 3.9+

## Quick Run

Run all protocols with assignment scenarios:

```bash
python main.py --protocol all --count 12 --size 16 --window 4 --timeout 8 --run-assignment-tests
```

Run only rdt 3.0 with a custom network:

```bash
python main.py --protocol rdt --count 10 --size 20 --timeout 8 --loss 0.2 --corrupt 0.1 --min-delay 1 --max-delay 5
```

Run only GBN:

```bash
python main.py --protocol gbn --count 20 --size 16 --window 5 --timeout 8 --loss 0.15 --corrupt 0.05 --min-delay 1 --max-delay 4
```

Run only SR:

```bash
python main.py --protocol sr --count 20 --size 16 --window 5 --timeout 8 --loss 0.15 --corrupt 0.05 --min-delay 1 --max-delay 4
```

## CLI Options

- `--protocol {rdt,gbn,sr,all}`
- `--count` number of packets/messages
- `--size` payload size of each packet
- `--window` sender window size (GBN/SR)
- `--timeout` fixed timeout for retransmission
- `--loss` packet loss probability in [0,1]
- `--corrupt` packet corruption probability in [0,1]
- `--min-delay`, `--max-delay` network delay bounds
- `--seed` random seed for reproducibility
- `--run-assignment-tests` run required 4 test scenarios
- `--verbose` print sender/receiver event trace

## Output

For each run, the simulator prints:

- protocol and scenario
- pass/fail (all packets delivered in order)
- events and simulated completion time
- sender packets and retransmissions
- network statistics (dropped/corrupted/scheduled packets and average delay)

## Files

- `main.py`: CLI runner and test scenario executor
- `protocols/common.py`: packet model, event loop, unreliable network model
- `protocols/rdt.py`: rdt 3.0 implementation
- `protocols/gbn.py`: Go-Back-N implementation
- `protocols/sr.py`: Selective Repeat implementation

