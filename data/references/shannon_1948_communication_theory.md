# A Mathematical Theory of Communication (1948)

**Author**: Claude E. Shannon  
**Source**: Bell System Technical Journal, Vol. 27, pp. 379–423, 623–656  

## Abstract
The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point. Frequently the messages have meaning; that is they refer to or are correlated according to some system with certain physical or conceptual entities. These semantic aspects of communication are irrelevant to the engineering problem.

## Discrete Representation of Information
A communication system consists essentially of five parts:
1. **Information Source**: Produces a message or sequence of messages to be communicated to the receiving terminal.
2. **Transmitter**: Operates on the message in some way to produce a signal suitable for transmission over the channel.
3. **Channel**: Merely the medium used to transmit the signal from transmitter to receiver.
4. **Receiver**: Performs the inverse operation of that done by the transmitter, reconstructing the message from the signal.
5. **Destination**: The person or thing for whom the message is intended.

## Measure of Information and Entropy
If the source produces a finite number of possible messages with probabilities $P_1, P_2, \dots, P_n$, the entropy $H$ is defined as:

$$H = -\sum_{i=1}^n P_i \log_2 P_i$$

This quantity measures how much information is produced on the average for each element of the message. The logarithmic base two gives units in bits (binary digits).

## Channel Capacity and Noise
The fundamental theorem for a discrete channel with noise states that if the source rate $R$ is less than the channel capacity $C$, there exists a coding system such that the output of the source can be transmitted over the channel with an arbitrarily small frequency of errors.
