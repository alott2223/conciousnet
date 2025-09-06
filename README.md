# ConsciousNet: Emergent Consciousness Neural Architecture

## Overview
ConsciousNet is a modular, extensible neural network architecture designed to simulate emergent qualities of consciousness. It features layered processing, persistent self-referential states, recursive feedback, meta-awareness, emotional/valuation weighting, a working self-model, and adaptive restructuring.

## Features
- **Sensory Layer:** Modular neural encoders for vision, text, and abstract signals.
- **Perception Layer:** Neural fusion of sensory modalities (concatenation, attention).
- **Narrative Layer:** Episodic, semantic, and procedural memory modules.
- **Meta-Cognitive Layer:** Neural critics, planners, and self-reflection routines.
- **Emotional/Valuation Module:** Multi-dimensional affective states and phi-like weighting.
- **Self-Model:** Tracks resources, strengths, weaknesses, goals, and predictions.
- **Feedback Recursion:** Bidirectional communication and reflection between layers.
- **Adaptive Restructuring:** Pruning, expansion, and rewriting of architecture in response to experience.
- **Advanced Logging:** Event replay, checkpointing, and model versioning.
- **Visualization:** Plot phi and affective state evolution.
- **Multi-Agent Support:** Run and compare multiple agents in parallel.
- **Interactive CLI:** Inspect, replay, and manipulate agent state.

## Installation
1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install torch matplotlib
   ```
3. Place `conscious_net.py` in your project directory.

## Usage
### CLI
Run the CLI:
```bash
python conscious_net.py
```

**Available commands:**
- `run` — Process 10 events and update the network
- `introspect` — Print current agent state
- `save` / `load` — Save/load persistent state
- `query [layer]` — Print state of a specific layer
- `tags` — List important event tags
- `get [tag]` — Retrieve event by tag
- `crisis` — Rewrite all layers (identity crisis)
- `overload` — Force memory overload and pruning
- `phi` — Visualize phi (significance) evolution
- `affect` — Visualize affective state evolution
- `multi [n]` — Create n agents
- `multi step` — Step all agents
- `multi introspect` — Introspect all agents
- `ckpt [name]` — Save a checkpoint
- `ckpt list` — List checkpoints
- `ckpt load [name]` — Load a checkpoint
- `replay [start] [end]` — Replay event history
- `event [idx]` — Print a specific event
- `clearlog` — Clear all logs
- `exit` — Quit

### Example: Using the Architecture in Python
```python
from conscious_net import ConsciousNet, DEFAULT_CONFIG, generate_input

# Create a ConsciousNet agent
net = ConsciousNet(DEFAULT_CONFIG)

# Process 20 events
for i in range(20):
    x = generate_input(i, DEFAULT_CONFIG)
    state = net.forward(x)
    net.feedback_recursion()
    net.adaptive_restructuring()

# Introspect and print the last 3 states
net.introspect()
net.print_log(3)

# Save a checkpoint
net.save_checkpoint('after_20_events')

# Replay the first 5 events
net.replay_events(0, 5)

# Visualize phi evolution (requires matplotlib)
phi_log = [s['emotion']['phi'] for s in net.log]
import matplotlib.pyplot as plt
plt.plot(phi_log)
plt.title('Phi Evolution')
plt.xlabel('Event')
plt.ylabel('Phi')
plt.show()
```

## Architecture Diagram
```
+-------------------+
|  Meta-Cognitive   |
|   Layer           |
+-------------------+
        ^   ^   ^
        |   |   |
        v   v   v
+-------------------+
|   Narrative Layer |
+-------------------+
        ^   ^   ^
        |   |   |
        v   v   v
+-------------------+
|  Perception Layer |
+-------------------+
        ^   ^   ^
        |   |   |
        v   v   v
+-------------------+
|   Sensory Layer   |
+-------------------+
        |
        v
+-------------------------+
|   Emotional/Valuation   |
|        Module           |
+-------------------------+
        |
        v
+-------------------------+
|     Self-Model         |
+-------------------------+
```

## Extending the System
- Add new sensory modalities by subclassing and plugging into the SensoryLayer.
- Implement new fusion strategies in PerceptionLayer.
- Add new critics, planners, or memory modules for richer meta-cognition.
- Use the API for scripting experiments, multi-agent scenarios, and custom visualizations.

## License
MIT License
