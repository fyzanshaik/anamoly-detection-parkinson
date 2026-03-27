# FAQ — Questions You Might Get Asked

Common questions about the project, the hardware choices, the ML model, and the design decisions. Answers are written to be explained verbally.

---

## The Project

**What does this project actually do?**
It detects faults in rotating machines — specifically DC motors — using vibration data. Two sensor nodes are attached to motors. They measure vibration using an accelerometer, run a neural network on the ESP32 itself to decide if the vibration is abnormal, and send results wirelessly to a gateway device that classifies the fault type and displays it on a screen and a web dashboard. No cloud, no internet, everything runs on the hardware.

**Why is this useful? What's the real-world application?**
This is predictive maintenance. In industry, motors and rotating machinery fail without warning — a bearing starts degrading weeks before it fails, but nobody knows until something breaks and production stops. Detecting the early vibration signature of a fault lets you schedule maintenance before it becomes a breakdown. This saves money and prevents safety incidents.

**Why did you use three devices instead of one?**
The architecture mirrors a real distributed system. In a real factory, sensors are physically at the machines and cannot all be wired back to one central device easily. Wireless sensor nodes are a natural design. Having a gateway separate from the sensors also means the gateway can monitor multiple machines simultaneously — the architecture scales. With one device you'd have a demo; with three you have a system.

**What faults does it detect?**
Three fault types:
- **Imbalance** — uneven mass distribution on the rotating shaft, causes periodic vibration at rotation frequency, raises all vibration axes proportionally
- **Bearing fault** — damage on the bearing surface causes sharp impulse events as the defect strikes rolling elements, mag_max spikes while mean stays moderate
- **Looseness** — mechanical looseness in the motor mounting or shaft causes irregular, random-amplitude vibration, mag_std increases unpredictably

Plus normal operation — the system correctly identifies when nothing is wrong.

---

## Why ESP32?

**Why not Arduino?**
An Arduino Uno has an 8-bit 16MHz CPU and 2KB RAM. The autoencoder alone needs to store and multiply 10×5 + 5×10 = 150 float weights. At 4 bytes per float that's 600 bytes just for weights, plus input/output buffers — already exceeds most of the Arduino's RAM. The ESP32 has a 240MHz dual-core CPU and 520KB RAM. Neural network inference on the edge requires a capable chip.

**Why not a Raspberry Pi?**
A Pi runs Linux, needs a minute to boot, costs $35–80, draws significant power, and is vastly overspecified for reading an accelerometer and running a ~1KB model. An ESP32 boots in under a second, costs $3, and draws milliamps. For an embedded sensing application, the ESP32 is the correct tool.

**Why not use the ESP32's Bluetooth instead of ESP-NOW?**
ESP-NOW is faster, simpler, and lower overhead for this use case. It operates at Layer 2 — there's no IP stack, no pairing process (just register a MAC address), and latency is sub-millisecond. Bluetooth on the ESP32 is fine for audio or file transfer but adds unnecessary complexity for a sensor that sends a 50-byte struct once per second.

---

## Why MPU6050?

**Why an accelerometer for detecting motor faults? Why not a microphone or temperature sensor?**
Vibration is the primary signature of mechanical faults in rotating machinery. Accelerometers directly measure vibration as acceleration, which is the industry-standard approach for predictive maintenance. A microphone picks up sound which is partially related but is affected by ambient noise and doesn't have the directional sensitivity of an accelerometer. Temperature is a lagging indicator — by the time a bearing gets hot enough to detect, it's already severely damaged. Vibration changes weeks before temperature.

**Why the MPU6050 specifically?**
It is inexpensive (~$1), widely available, well-documented, and has a mature Arduino/ESP32 library. It communicates over I2C which only needs 2 wires. For the purposes of this project, its ±8G range and 21Hz digital low-pass filter are well matched to the vibration frequencies of small DC motors. It also has a built-in gyroscope that isn't used here but demonstrates the sensor is capable of more.

**What is the ±8G range setting?**
The MPU6050 can be configured to measure different acceleration ranges: ±2G, ±4G, ±8G, or ±16G. A smaller range gives finer resolution (more ADC bits per unit of acceleration). ±8G was chosen because it covers both normal motor vibration (well under 1G) and strong fault vibration without saturating the sensor.

**What is the 21Hz low-pass filter?**
The MPU6050 has a built-in digital low-pass filter that removes high-frequency noise from the accelerometer output. 21Hz means frequencies above 21Hz are attenuated. This filters out electrical interference from the motor driver's PWM signal (which operates at much higher frequency) while preserving the mechanical vibration frequencies of the motor shaft rotation, which are typically 5–20Hz at the speeds used.

---

## The Features

**Why 10 features? Why not just send raw samples?**
Raw data from 64 samples at 3 axes = 192 floats = 768 bytes. The ESP-NOW max payload is 250 bytes. More importantly, raw time-series data requires more complex models (CNNs, RNNs) to process. Statistical features extracted from the window are compact (10 floats = 40 bytes), preserve the physically meaningful information, and work with simple feedforward networks. This is called feature engineering.

**Why 64 samples?**
It's a power of 2 (convenient for buffer indexing), and at 10ms per sample gives a 640ms window. This is long enough to:
- Capture multiple rotation cycles at the motor speeds used
- Compute statistically stable mean and standard deviation estimates
- Detect transient events like bearing impulse patterns

**Why is mag_max the key feature for bearing faults?**
A bearing fault creates a physical impact every time the defect passes over a rolling element. This produces a brief, sharp spike in acceleration — the magnitude spikes momentarily but the average vibration level doesn't necessarily increase much. mag_max captures the worst-case impulse in the window; the ratio of mag_max to mag_mean (called the impulse factor or crest factor) is the classic bearing fault indicator in vibration analysis.

**Why do axis means contribute almost nothing to detection?**
Faults change how much the motor vibrates — the amplitude and character of motion. They don't systematically push the motor's center position. Mean acceleration in each axis reflects the static orientation of the sensor under gravity and the mean force of steady motor operation, neither of which changes meaningfully with fault type. Standard deviations and magnitude features capture dynamic behavior.

---

## The Machine Learning

**What is an autoencoder and why use it for anomaly detection?**
An autoencoder is a neural network that learns to compress input data into a smaller representation and then reconstruct it back to the original. The key insight for anomaly detection: train it only on normal data. After training, the network has learned the internal structure of normal vibration. When you feed it a normal sample, it reconstructs it accurately. When you feed it a fault sample — which looks different from anything it was trained on — it fails to reconstruct it accurately and produces a high error. That reconstruction error is the anomaly score.

**Why not train a direct classifier for anomaly detection?**
A classifier requires labeled examples of every fault type you want to detect. In the real world, you often have abundant data from machines running normally and very few examples of actual failures (you don't want your machines to fail). An autoencoder only requires normal data to train, which is always available. It also generalizes to fault types it has never seen — if a new failure mode appears that produces unusual vibration, the reconstruction error will be high even if that fault type was never in the training set.

**What is the threshold and how was it chosen?**
The threshold is 0.966 — the 95th percentile of reconstruction errors on the normal training data. This means 95% of normal readings produce a score below 0.966. If a current reading exceeds that, there is less than 5% chance it's normal. Choosing the 95th percentile is a deliberate trade-off: it gives a 5.2% false positive rate (normal readings occasionally flagged as anomalous) in exchange for 100% detection rate (no faults missed). In predictive maintenance, a missed fault is far more costly than a false alarm, so this trade-off is appropriate.

**What does the neural classifier on the gateway do and why is it separate?**
The sensor nodes detect that something is wrong (anomaly score). The gateway then classifies what is wrong (fault type). This separation makes sense architecturally: anomaly detection is unsupervised and runs on the sensor node in real-time; fault classification is supervised and runs on the gateway which has more context (it can see both nodes simultaneously and has the full feature vector for XAI). The classifier is a standard feedforward network trained with labeled examples of each fault type.

**What is 97.9% accuracy? What are the 2.1% of errors?**
The classifier correctly identifies the fault type 979 times out of 1000. The errors are almost entirely Imbalance ↔ Bearing confusions (~15-17 samples each way out of 400 per class). This is physically expected: both faults raise vibration magnitude. The difference is the impulsive vs proportional pattern, which is subtle enough that occasional confusion happens. Looseness is nearly perfect (399/400) because its random amplitude pattern is distinct.

**Why did you implement the neural network from scratch in numpy instead of using TensorFlow or Keras?**
Python 3.14 (the version used at time of development) broke TensorFlow and PyTorch compatibility. But more importantly, a neural network is fundamentally just matrix multiplication and gradient descent — the chain rule of calculus applied to a computational graph. Implementing it from scratch demonstrates actual understanding of how neural networks work, rather than calling `.fit()` on a framework and treating it as a black box. The math is the same.

**Why only 299 parameters total?**
The network is deliberately small. The problem — classifying 10 features into 4 categories — does not require millions of parameters. More parameters would overfit on the small training set and would also be unnecessarily large for an embedded system. 299 floats × 4 bytes = 1196 bytes. The entire two-model system fits in 1KB, which is less than a single JPEG thumbnail.

---

## The Rule Classifier

**Why have a rule-based classifier alongside the neural network?**
Two reasons:
1. **Confidence estimation:** When both classifiers agree, confidence is HIGH. When they disagree, it signals ambiguity. A single classifier has no way to generate this signal.
2. **Interpretability:** The rule classifier's logic is directly explainable — "impulse ratio exceeded 1.08 while max was 1.5× the mean." This gives a physical justification for the classification that the neural network alone cannot provide. This is the XAI (Explainable AI) component.

**What is the impulse ratio?**
`mag_max / mag_mean` — the ratio of the peak magnitude to the average magnitude in the window. For smooth vibration (imbalance), this ratio is close to 1. For bearing faults with sharp spikes, this ratio is significantly above 1. A value above 1.08 is the threshold used by the rule classifier.

---

## Hardware Connectivity

**Why is boot order important?**
ESP-NOW requires each device to be initialized and know its peers' MAC addresses before packets can be sent. The gateway must be running first so when the sensor nodes boot and register the gateway MAC as a peer, the gateway is already listening. If a sensor node boots before the gateway, the first few packets may be lost, and the gateway won't have Node state until it receives its first packet anyway. Boot order: Gateway → Node 1 → Node 2.

**Why does Node 2 wait 25 seconds before starting?**
To offset the 50-second cycles so the two nodes are never in ANOMALY phase simultaneously. If both nodes fault at the same time, the demo and LCD become confusing — you can't see one fault resolve while another continues. With a 25-second offset (half the cycle length), when Node 1 is in ANOMALY phase, Node 2 is in NORMAL phase, and vice versa. This makes the demonstration significantly clearer.

**What does "R / S / D" on the dashboard mean?**
Data quality indicator:
- **R (Real):** The MPU6050 is connected and returning real accelerometer readings
- **S (Synthetic):** The MPU6050 is not detected at boot; the node is generating feature values mathematically from motor speed and fault patterns
- **D (Degraded):** The MPU6050 was connected but dropped mid-operation; the node is using the last known real reading as a base with slight updates

**Why does the green LED also turn on the onboard blue LED?**
The ESP32 NodeMCU-32S onboard blue LED is hardwired to GPIO2. The green LED is also connected to GPIO2. When GPIO2 goes HIGH, both light simultaneously. This is a physical characteristic of the board, not a bug.

---

## The Dashboard

**Why Server-Sent Events (SSE) and not WebSockets?**
SSE is simpler for this use case. It's a one-way stream from server to browser — the gateway pushes updates, the browser just displays them. WebSockets are bidirectional and add complexity that isn't needed. SSE auto-reconnects if the connection drops, it works over standard HTTP, and the `EventSource` API in the browser handles it natively.

**Why does the gateway serve the dashboard itself instead of a separate server?**
The project goal is fully self-contained edge AI — no internet, no cloud, no external services. The ESP32 runs an AsyncWebServer that serves the HTML page and the SSE stream. Anyone on the local WiFi (AnomalyNet) can connect. There is no dependency on any external infrastructure.

**What is the ⚡ indicator on the dashboard?**
A lightning bolt appears on a node's status card when `phase == NORMAL && isAnomalous == true`. This specific combination means the anomaly was not triggered by the scripted fault cycle — instead, the autoencoder or Z-score genuinely fired on a reading during the normal operating phase. This is the ML model catching something on its own, separate from the pre-programmed fault injection.

---

## Evaluation and Results

**What does 100% detection rate mean?**
Every single fault sample in the test set was correctly identified as anomalous by the autoencoder. None were missed. This is the false negative rate = 0%.

**What does 5.2% false positive rate mean?**
Of all normal samples, 5.2% were incorrectly flagged as anomalous. This is by design — the 95th percentile threshold accepts this level of false alarms in exchange for zero missed faults.

**Is 5.2% FPR acceptable?**
For predictive maintenance, yes. The consequence of a false positive is an unnecessary inspection — you check the machine and find nothing wrong. The consequence of a false negative is an undetected fault progressing to failure — equipment damage, production downtime, possibly a safety incident. The cost ratio makes a bias toward false positives the right choice.

**Could you reduce the false positive rate?**
Yes, by raising the threshold. If you set the threshold at the 99th percentile instead of the 95th, the FPR would drop to ~1% but some faults near the boundary might be missed. The threshold is a tunable parameter — the current setting was chosen to prioritize detection completeness.
