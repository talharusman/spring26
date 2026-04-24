# Report: Reliable Data Transfer Protocols

## Student Details

- Student ID: (fill here)
- Name: (fill here)
- Section: (fill here)

## Objective

Implement and test:

- rdt 3.0 (Stop-and-Wait)
- Go-Back-N (GBN)
- Selective Repeat (SR)

The channel is unreliable and may drop packets, corrupt packets, or delay packets.

## Design Summary

- A discrete-event simulator is used.
- Both data and ACK packets pass through the same unreliable network model.
- Sender and receiver logic for each protocol follows finite-state-machine behavior.
- Sequence numbers are used to ensure reliable in-order delivery.
- Packet size and packet count are configurable from the command line.

## FSM Diagrams

### rdt 3.0 Sender

```mermaid
stateDiagram-v2
    [*] --> WaitCall0
    WaitCall0 --> WaitAck0: send pkt(seq=0), start timer
    WaitAck0 --> WaitCall1: rcv ACK0 (not corrupt), stop timer
    WaitAck0 --> WaitAck0: timeout / retransmit seq0
    WaitAck0 --> WaitAck0: corrupt ACK / ignore

    WaitCall1 --> WaitAck1: send pkt(seq=1), start timer
    WaitAck1 --> WaitCall0: rcv ACK1 (not corrupt), stop timer
    WaitAck1 --> WaitAck1: timeout / retransmit seq1
    WaitAck1 --> WaitAck1: corrupt ACK / ignore
```

### rdt 3.0 Receiver

```mermaid
stateDiagram-v2
    [*] --> Wait0
    Wait0 --> Wait1: rcv pkt0 (not corrupt), deliver, send ACK0
    Wait0 --> Wait0: corrupt pkt or pkt1, resend last ACK

    Wait1 --> Wait0: rcv pkt1 (not corrupt), deliver, send ACK1
    Wait1 --> Wait1: corrupt pkt or pkt0, resend last ACK
```

### GBN Sender

```mermaid
stateDiagram-v2
    [*] --> WindowSend
    WindowSend --> WindowSend: send while nextseq < base + N
    WindowSend --> WaitEvents
    WaitEvents --> WindowSend: rcv cumulative ACK, slide base
    WaitEvents --> WaitEvents: timeout(base) / retransmit all [base..nextseq-1]
    WaitEvents --> WaitEvents: corrupt ACK / ignore
```

### GBN Receiver

```mermaid
stateDiagram-v2
    [*] --> ExpectK
    ExpectK --> ExpectK: rcv expected pkt k, deliver, ACK k, k++
    ExpectK --> ExpectK: out-of-order/corrupt packet, discard, ACK last in-order
```

### SR Sender

```mermaid
stateDiagram-v2
    [*] --> WindowSend
    WindowSend --> WindowSend: send while nextseq < base + N
    WindowSend --> WaitEvents
    WaitEvents --> WaitEvents: rcv ACK(i), mark i acked
    WaitEvents --> WindowSend: slide base over consecutive acked seq
    WaitEvents --> WaitEvents: timeout(i) / retransmit only pkt i
```

### SR Receiver

```mermaid
stateDiagram-v2
    [*] --> ReceiveWindow
    ReceiveWindow --> ReceiveWindow: pkt in window and valid -> buffer + ACK(seq)
    ReceiveWindow --> ReceiveWindow: pkt < base -> duplicate ACK(seq)
    ReceiveWindow --> ReceiveWindow: pkt > window -> ignore
    ReceiveWindow --> ReceiveWindow: if base buffered, deliver in-order and advance base
```

## Testing Scenarios

The simulator includes these required scenarios:

1. No packet loss/corruption
2. Packet loss
3. Packet corruption
4. Delayed packets

Command used:

```bash
python main.py --protocol all --count 12 --size 16 --window 4 --timeout 8 --run-assignment-tests
```

## Observations

- rdt 3.0 works reliably but has lower throughput since only one packet is in-flight.
- GBN improves throughput using pipelining, but timeout causes retransmission of an entire suffix of the window.
- SR improves retransmission efficiency by resending only timed-out packets and buffering out-of-order packets at the receiver.
- In all tested scenarios, each protocol recovers and eventually delivers all packets in order.

## How to Run

See `README.md` for commands and options.
