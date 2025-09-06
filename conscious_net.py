import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os
import random
import sys
import time
from typing import Dict, Any, List, Optional, Callable
import matplotlib.pyplot as plt
import datetime

class VisionEncoder(nn.Module):
    """Simple CNN-based encoder for vision input."""
    def __init__(self, input_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * input_dim * input_dim, out_dim)
        self.input_dim = input_dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    """Simple RNN-based encoder for text input."""
    def __init__(self, vocab_size: int = 100, embed_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, out_dim, batch_first=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        _, h = self.rnn(x)
        return h.squeeze(0)

class AbstractEncoder(nn.Module):
    """Simple MLP encoder for abstract signals."""
    def __init__(self, input_dim: int = 8, out_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SensoryLayer:
    """
    SensoryLayer processes raw input from multiple modalities using neural encoders.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {'vision_dim': 32, 'text_vocab': 100, 'abstract_dim': 8, 'out_dim': 16}
        self.vision_encoder = VisionEncoder(self.config['vision_dim'], self.config['out_dim'])
        self.text_encoder = TextEncoder(self.config['text_vocab'], 16, self.config['out_dim'])
        self.abstract_encoder = AbstractEncoder(self.config['abstract_dim'], self.config['out_dim'])
        self.vision_count = 0
        self.text_count = 0
        self.abstract_count = 0
    def encode_vision(self, vision: torch.Tensor) -> torch.Tensor:
        self.vision_count += 1
        return self.vision_encoder(vision)
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        self.text_count += 1
        return self.text_encoder(text)
    def encode_abstract(self, abstract: torch.Tensor) -> torch.Tensor:
        self.abstract_count += 1
        return self.abstract_encoder(abstract)
    def forward(self, x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        vision = self.encode_vision(x['vision'])
        text = self.encode_text(x['text'])
        abstract = self.encode_abstract(x['abstract'])
        return {'vision': vision, 'text': text, 'abstract': abstract}
    def prune(self):
        self.vision_count = max(0, self.vision_count - 1)
        self.text_count = max(0, self.text_count - 1)
        self.abstract_count = max(0, self.abstract_count - 1)
    def expand(self):
        # Placeholder: could add more channels or layers
        pass
    def rewrite(self):
        self.vision_count = 0
        self.text_count = 0
        self.abstract_count = 0
    def report(self):
        return {
            'vision_count': self.vision_count,
            'text_count': self.text_count,
            'abstract_count': self.abstract_count,
            'config': self.config
        }

class PerceptionFusion(nn.Module):
    """Fusion module for combining sensory embeddings."""
    def __init__(self, in_dim: int = 16, out_dim: int = 32, mode: str = 'concat'):
        super().__init__()
        self.mode = mode
        if mode == 'concat':
            self.fc = nn.Linear(in_dim * 3, out_dim)
        elif mode == 'attention':
            self.attn = nn.MultiheadAttention(in_dim, num_heads=2, batch_first=True)
            self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, vision: torch.Tensor, text: torch.Tensor, abstract: torch.Tensor) -> torch.Tensor:
        if self.mode == 'concat':
            x = torch.cat([vision, text, abstract], dim=-1)
            return self.fc(x)
        elif self.mode == 'attention':
            # Stack as sequence: [vision, text, abstract]
            seq = torch.stack([vision, text, abstract], dim=1)
            attn_out, _ = self.attn(seq, seq, seq)
            pooled = attn_out.mean(dim=1)
            return self.fc(pooled)
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")

class PerceptionLayer:
    """
    PerceptionLayer fuses sensory embeddings into a structured representation.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {'in_dim': 16, 'out_dim': 32, 'fusion_mode': 'concat'}
        self.fusion = PerceptionFusion(self.config['in_dim'], self.config['out_dim'], self.config['fusion_mode'])
        self.fusion_history: List[torch.Tensor] = []
    def fuse(self, sensory_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        fused = self.fusion(sensory_dict['vision'], sensory_dict['text'], sensory_dict['abstract'])
        self.fusion_history.append(fused.detach().cpu())
        return {'fused': fused, 'fusion_history': self.fusion_history[-5:]}
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.fuse(x)
    def prune(self):
        if len(self.fusion_history) > 10:
            self.fusion_history = self.fusion_history[-10:]
    def expand(self):
        # Placeholder: could switch fusion mode or add more heads
        pass
    def rewrite(self):
        self.fusion_history = []
    def report(self):
        return {
            'fusion_history': [f.tolist() for f in self.fusion_history[-5:]],
            'config': self.config
        }

class EpisodicMemory(nn.Module):
    """Neural episodic memory using a GRU and external memory buffer."""
    def __init__(self, input_dim: int = 32, mem_size: int = 100):
        super().__init__()
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.mem_size = mem_size
        self.memory: List[torch.Tensor] = []
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store in memory
        if len(self.memory) >= self.mem_size:
            self.memory.pop(0)
        self.memory.append(x.detach().cpu())
        # Return last hidden state
        x_seq = torch.stack(self.memory[-min(len(self.memory), 10):]).unsqueeze(0)
        _, h = self.gru(x_seq)
        return h.squeeze(0)
    def retrieve(self, idx: int) -> Optional[torch.Tensor]:
        if 0 <= idx < len(self.memory):
            return self.memory[idx]
        return None
    def clear(self):
        self.memory = []

class SemanticMemory(nn.Module):
    """Neural semantic memory using a key-value store and MLP."""
    def __init__(self, key_dim: int = 32, value_dim: int = 32, mem_size: int = 100):
        super().__init__()
        self.keys: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.mem_size = mem_size
        self.mlp = nn.Sequential(
            nn.Linear(key_dim, 64),
            nn.ReLU(),
            nn.Linear(64, value_dim)
        )
    def store(self, key: torch.Tensor, value: torch.Tensor):
        if len(self.keys) >= self.mem_size:
            self.keys.pop(0)
            self.values.pop(0)
        self.keys.append(key.detach().cpu())
        self.values.append(value.detach().cpu())
    def retrieve(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.keys:
            return None
        sims = [F.cosine_similarity(query, k, dim=0) for k in self.keys]
        idx = int(torch.argmax(torch.tensor(sims)))
        return self.values[idx]
    def clear(self):
        self.keys = []
        self.values = []

class ProceduralMemory(nn.Module):
    """Procedural memory as a simple policy network."""
    def __init__(self, input_dim: int = 32, out_dim: int = 8):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

class NarrativeLayer:
    """
    NarrativeLayer maintains episodic, semantic, and procedural memory.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {'input_dim': 32, 'mem_size': 100}
        self.episodic = EpisodicMemory(self.config['input_dim'], self.config['mem_size'])
        self.semantic = SemanticMemory(self.config['input_dim'], self.config['input_dim'], self.config['mem_size'])
        self.procedural = ProceduralMemory(self.config['input_dim'], 8)
        self.event_tags: List[str] = []
        self.event_count = 0
    def add_to_memory(self, fused: torch.Tensor) -> Dict[str, Any]:
        tag = f"event_{self.event_count}"
        self.event_count += 1
        epi = self.episodic(fused)
        self.semantic.store(fused, epi)
        proc = self.procedural(fused)
        self.event_tags.append(tag)
        return {'episodic': epi, 'semantic': epi, 'procedural': proc, 'tags': list(self.event_tags)}
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return self.add_to_memory(x['fused'])
    def prune(self):
        self.episodic.memory = self.episodic.memory[-10:]
        self.semantic.keys = self.semantic.keys[-10:]
        self.semantic.values = self.semantic.values[-10:]
        self.event_tags = self.event_tags[-10:]
    def expand(self):
        self.config['mem_size'] += 10
        self.episodic.mem_size = self.config['mem_size']
        self.semantic.mem_size = self.config['mem_size']
    def rewrite(self):
        self.episodic.clear()
        self.semantic.clear()
        self.event_tags = []
        self.event_count = 0
    def tag_important_events(self, phi_history: List[float]) -> List[str]:
        tags = []
        for i, phi in enumerate(phi_history):
            if phi > 0.8 and i < len(self.event_tags):
                tags.append(self.event_tags[i])
        return tags
    def retrieve_event(self, tag: str) -> Optional[torch.Tensor]:
        if tag in self.event_tags:
            idx = self.event_tags.index(tag)
            return self.episodic.memory[idx] if idx < len(self.episodic.memory) else None
        return None
    def report(self):
        return {
            'episodic_size': len(self.episodic.memory),
            'semantic_size': len(self.semantic.keys),
            'procedural': 'policy',
            'recent_tags': self.event_tags[-5:],
            'config': self.config
        }

class Critic(nn.Module):
    """Meta-cognitive critic network."""
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))

class Planner(nn.Module):
    """Meta-cognitive planner network."""
    def __init__(self, input_dim: int = 32, out_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(input_dim, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class MetaCognitiveLayer:
    """
    MetaCognitiveLayer monitors, critiques, and plans using neural critics and planners.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {'input_dim': 32}
        self.critics = [Critic(self.config['input_dim']) for _ in range(2)]
        self.planners = [Planner(self.config['input_dim'], 8) for _ in range(2)]
        self.critique_log: List[str] = []
        self.last_timeline_length = 0
        self.last_event = None
    def monitor(self, timeline: List[Any]) -> Dict[str, Any]:
        self.last_timeline_length = len(timeline)
        self.last_event = timeline[-1] if timeline else None
        critique = self.critique(timeline)
        plan = self.plan(timeline)
        return {'timeline_length': self.last_timeline_length, 'last_event': self.last_event, 'critique': critique, 'plan': plan}
    def critique(self, timeline: List[Any]) -> str:
        if not timeline:
            msg = 'No events to critique.'
        else:
            # Use critics to score last event
            x = timeline[-1] if timeline else torch.zeros(self.config['input_dim'])
            scores = [float(c(torch.tensor(x).float().unsqueeze(0))) for c in self.critics]
            msg = f'Critic scores: {scores}'
        self.critique_log.append(msg)
        return msg
    def plan(self, timeline: List[Any]) -> str:
        if not timeline:
            return 'No plan.'
        x = timeline[-1] if timeline else torch.zeros(self.config['input_dim'])
        plans = [p(torch.tensor(x).float().unsqueeze(0)).tolist() for p in self.planners]
        return f'Plans: {plans}'
    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return self.monitor(x['episodic'] if 'episodic' in x else [])
    def prune(self):
        self.critique_log = self.critique_log[-10:]
    def expand(self):
        self.critics.append(Critic(self.config['input_dim']))
        self.planners.append(Planner(self.config['input_dim'], 8))
    def rewrite(self):
        self.critics = [Critic(self.config['input_dim']) for _ in range(2)]
        self.planners = [Planner(self.config['input_dim'], 8) for _ in range(2)]
        self.critique_log = []
    def influence_perception(self, perception_layer):
        if self.last_timeline_length % 7 == 0:
            perception_layer.expand()
    def report(self):
        return {
            'last_timeline_length': self.last_timeline_length,
            'last_event': str(self.last_event),
            'critique_log': self.critique_log[-5:],
            'num_critics': len(self.critics),
            'num_planners': len(self.planners),
            'config': self.config
        }

class EmotionalValuationModule:
    """
    EmotionalValuationModule tracks multi-dimensional affective states.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {'dims': 3, 'novelty_weight': 1.0, 'repeat_weight': 0.1}
        self.affect = torch.zeros(self.config['dims'])
        self.phi_log: List[float] = []
    def assign_phi(self, last_event: Any) -> Dict[str, Any]:
        # Multi-dimensional affect: [novelty, valence, arousal]
        novelty = random.random()
        valence = random.uniform(-1, 1)
        arousal = random.random()
        self.affect = torch.tensor([novelty, valence, arousal])
        phi = float(novelty * self.config['novelty_weight'] + arousal * 0.5)
        self.phi_log.append(phi)
        return {'phi': phi, 'affect': self.affect.tolist(), 'event': last_event, 'phi_log': list(self.phi_log)}
    def forward(self, x: Dict[str, Any]) -> Dict[str, Any]:
        return self.assign_phi(x['last_event'])
    def prune(self):
        self.phi_log = self.phi_log[-10:]
    def expand(self):
        self.config['dims'] += 1
        self.affect = torch.zeros(self.config['dims'])
    def rewrite(self):
        self.phi_log = []
        self.affect = torch.zeros(self.config['dims'])
    def report(self):
        return {
            'phi_log': self.phi_log[-5:],
            'affect': self.affect.tolist(),
            'config': self.config
        }

class SelfModel:
    """
    SelfModel tracks resources, goals, and self-prediction.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.state = {
            'event_count': 0,
            'phi_history': [],
            'strengths': [],
            'weaknesses': [],
            'resources': {'memory': 0},
            'goals': [],
            'predictions': []
        }
        self.config = config or {'track_strengths': True, 'track_weaknesses': True}
    def update(self, x: Dict[str, Any]):
        self.state['event_count'] += 1
        self.state['phi_history'].append(x.get('phi', 0))
        if self.config['track_strengths'] and x.get('phi', 0) > 0.5:
            self.state['strengths'].append(f"event_{self.state['event_count']}")
        if self.config['track_weaknesses'] and x.get('phi', 0) < 0.2:
            self.state['weaknesses'].append(f"event_{self.state['event_count']}")
        self.state['resources']['memory'] = len(self.state['phi_history'])
        # Dummy goal and prediction logic
        if self.state['event_count'] % 5 == 0:
            self.state['goals'].append(f"goal_{self.state['event_count']}")
            self.state['predictions'].append(f"pred_{self.state['event_count']}")
    def get_state(self) -> Dict[str, Any]:
        return self.state
    def prune(self):
        self.state['phi_history'] = self.state['phi_history'][-10:]
        self.state['strengths'] = self.state['strengths'][-10:]
        self.state['weaknesses'] = self.state['weaknesses'][-10:]
        self.state['goals'] = self.state['goals'][-5:]
        self.state['predictions'] = self.state['predictions'][-5:]
    def expand(self):
        self.state['resources']['memory'] += 10
    def rewrite(self):
        self.state = {
            'event_count': 0,
            'phi_history': [],
            'strengths': [],
            'weaknesses': [],
            'resources': {'memory': 0},
            'goals': [],
            'predictions': []
        }
    def report(self) -> Dict[str, Any]:
        return self.state
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.state, f)
    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.state = pickle.load(f)

class ConsciousNet:
    def __init__(self, config=None):
        self.sensory = SensoryLayer(config.get('sensory') if config else None)
        self.perception = PerceptionLayer(config.get('perception') if config else None)
        self.narrative = NarrativeLayer(config.get('narrative') if config else None)
        self.meta = MetaCognitiveLayer(config.get('meta') if config else None)
        self.emotion = EmotionalValuationModule(config.get('emotion') if config else None)
        self.self_model = SelfModel(config.get('self_model') if config else None)
        self.log = []
        self.event_log = []  # For advanced logging and replay
        self.checkpoints = {}
    def forward(self, x):
        sensory_out = self.sensory.forward(x)
        perception_out = self.perception.forward(sensory_out)
        narrative_out = self.narrative.forward(perception_out)
        meta_out = self.meta.forward(narrative_out)
        emotion_out = self.emotion.forward(meta_out)
        self.self_model.update(emotion_out)
        state = {
            'timestamp': datetime.datetime.now().isoformat(),
            'input': x,
            'sensory': sensory_out,
            'perception': perception_out,
            'narrative': narrative_out,
            'meta': meta_out,
            'emotion': emotion_out,
            'self_model': self.self_model.get_state()
        }
        self.log.append(state)
        self.event_log.append(state)
        return state
    def save_checkpoint(self, name=None):
        """Save a checkpoint of the current agent state."""
        if not name:
            name = f"ckpt_{len(self.checkpoints)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoints[name] = {
            'log': list(self.log),
            'event_log': list(self.event_log),
            'self_model': self.self_model.get_state().copy()
        }
        print(f"Checkpoint '{name}' saved.")
    def load_checkpoint(self, name):
        """Restore agent state from a checkpoint."""
        if name in self.checkpoints:
            ckpt = self.checkpoints[name]
            self.log = list(ckpt['log'])
            self.event_log = list(ckpt['event_log'])
            self.self_model.state = ckpt['self_model'].copy()
            print(f"Checkpoint '{name}' loaded.")
        else:
            print(f"Checkpoint '{name}' not found.")
    def list_checkpoints(self):
        return list(self.checkpoints.keys())
    def replay_events(self, start=0, end=None):
        """Replay events from the event log."""
        end = end or len(self.event_log)
        for i, event in enumerate(self.event_log[start:end]):
            print(f"Event {i+start}: {event['timestamp']} | Input: {event['input']}")
    def print_event(self, idx):
        if 0 <= idx < len(self.event_log):
            event = self.event_log[idx]
            print(f"Event {idx}: {event['timestamp']}\nInput: {event['input']}\nState: {event['self_model']}")
        else:
            print(f"Event {idx} not found.")
    def clear_log(self):
        self.log = []
        self.event_log = []
        print("Logs cleared.")
    def feedback_recursion(self):
        # Meta can influence perception
        self.meta.influence_perception(self.perception)
        # If critique is negative, narrative prunes
        if self.meta.critique_log and 'No events' in self.meta.critique_log[-1]:
            self.narrative.prune()
    def adaptive_restructuring(self):
        state = self.self_model.get_state()
        avg_phi = sum(state['phi_history']) / len(state['phi_history']) if state['phi_history'] else 0
        if avg_phi < 0.5:
            self.sensory.expand()
            self.perception.expand()
            self.narrative.expand()
            self.meta.expand()
            self.emotion.expand()
            self.self_model.expand()
        if state['event_count'] > 20:
            self.sensory.prune()
            self.perception.prune()
            self.narrative.prune()
            self.meta.prune()
            self.emotion.prune()
            self.self_model.prune()
        if avg_phi > 0.9:
            self.sensory.rewrite()
            self.perception.rewrite()
            self.narrative.rewrite()
            self.meta.rewrite()
            self.emotion.rewrite()
            self.self_model.rewrite()
    def introspect(self):
        print("\n--- ConsciousNet Introspection ---")
        print(f"Sensory config: {self.sensory.config}")
        print(f"Perception config: {self.perception.config}")
        print(f"Narrative memory: {len(self.narrative.memory)} events")
        print(f"Meta critique log: {self.meta.critique_log[-3:]}")
        print(f"Emotion phi log: {self.emotion.phi_log[-3:]}")
        print(f"Self-model state: {self.self_model.get_state()}")
        print("----------------------------------\n")
    def print_log(self, n=5):
        print(f"\n--- Last {n} States ---")
        for state in self.log[-n:]:
            print(state)
        print("----------------------\n")
    def save_state(self, path='consciousnet_state.pkl'):
        state = {
            'self_model': self.self_model.state,
            'narrative_memory': self.narrative.memory,
            'narrative_tags': self.narrative.event_tags
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    def load_state(self, path='consciousnet_state.pkl'):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                state = pickle.load(f)
                self.self_model.state = state.get('self_model', self.self_model.state)
                self.narrative.memory = state.get('narrative_memory', self.narrative.memory)
                self.narrative.event_tags = state.get('narrative_tags', self.narrative.event_tags)
    def query_layer(self, layer_name):
        if layer_name == 'sensory':
            return self.sensory.report()
        if layer_name == 'perception':
            return self.perception.report()
        if layer_name == 'narrative':
            return self.narrative.report()
        if layer_name == 'meta':
            return self.meta.report()
        if layer_name == 'emotion':
            return self.emotion.report()
        if layer_name == 'self_model':
            return self.self_model.report()
        return None
    def tag_important_events(self):
        return self.narrative.tag_important_events(self.self_model.state['phi_history'])
    def retrieve_event(self, tag):
        return self.narrative.retrieve_event(tag)

def preprocess_vision(raw: Any, input_dim: int = 32) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw
    return torch.rand(1, input_dim, input_dim)

def preprocess_text(raw: Any, vocab_size: int = 100, seq_len: int = 10) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw
    return torch.randint(0, vocab_size, (seq_len,))

def preprocess_abstract(raw: Any, input_dim: int = 8) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw
    return torch.rand(input_dim)

def generate_input(event_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'vision': preprocess_vision(None, config['sensory']['vision_dim']),
        'text': preprocess_text(None, config['sensory']['text_vocab']),
        'abstract': preprocess_abstract(None, config['sensory']['abstract_dim'])
    }

def visualize_phi(phi_log: List[float]):
    plt.figure(figsize=(8, 3))
    plt.plot(phi_log, label='Phi (Significance)')
    plt.xlabel('Event')
    plt.ylabel('Phi')
    plt.title('Phi Evolution Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_affect(affect_log: List[List[float]]):
    plt.figure(figsize=(8, 3))
    affect_arr = torch.tensor(affect_log)
    for i in range(affect_arr.shape[1]):
        plt.plot(affect_arr[:, i], label=f'Affect Dim {i}')
    plt.xlabel('Event')
    plt.ylabel('Affect Value')
    plt.title('Affective State Evolution')
    plt.legend()
    plt.tight_layout()
    plt.show()

class MultiAgentSystem:
    def __init__(self, num_agents: int, config: Dict[str, Any]):
        self.agents = [ConsciousNet(config) for _ in range(num_agents)]
        self.num_agents = num_agents
        self.config = config
    def step_all(self):
        for i, agent in enumerate(self.agents):
            x = generate_input(agent.self_model.state['event_count'], self.config)
            agent.forward(x)
            agent.feedback_recursion()
            agent.adaptive_restructuring()
    def introspect_all(self):
        for i, agent in enumerate(self.agents):
            print(f"\n--- Agent {i} ---")
            agent.introspect()

# Configuration system
DEFAULT_CONFIG = {
    'sensory': {'vision_dim': 8, 'text_dim': 8, 'abstract_dim': 8},
    'perception': {'fusion_mode': 'concat'},
    'narrative': {'max_memory': 100},
    'meta': {'monitor_mode': 'basic'},
    'emotion': {'novelty_weight': 1.0, 'repeat_weight': 0.1},
    'self_model': {'track_strengths': True, 'track_weaknesses': True}
}

# Input simulation
def run_demo(steps=30, introspect_every=10):
    net = ConsciousNet(DEFAULT_CONFIG)
    for i in range(steps):
        x = generate_input(i, DEFAULT_CONFIG)
        state = net.forward(x)
        net.feedback_recursion()
        net.adaptive_restructuring()
        if (i+1) % introspect_every == 0:
            net.introspect()
            net.print_log(2)
    print("Demo complete.")
    net.introspect()
    net.print_log(5)

# CLI interface
def main():
    print("ConsciousNet Demo CLI")
    print("Type 'run', 'introspect', 'save', 'load', 'query [layer]', 'tags', 'get [tag]', 'crisis', 'overload', 'phi', 'affect', 'multi [n]', 'multi step', 'multi introspect', 'ckpt [name]', 'ckpt list', 'ckpt load [name]', 'replay [start] [end]', 'event [idx]', 'clearlog', or 'exit'.")
    net = ConsciousNet(DEFAULT_CONFIG)
    event_id = 0
    affect_log = []
    phi_log = []
    multi_agents = None
    while True:
        cmd = input('> ').strip().lower()
        if cmd == 'run':
            for _ in range(10):
                x = generate_input(event_id, DEFAULT_CONFIG)
                state = net.forward(x)
                net.feedback_recursion()
                net.adaptive_restructuring()
                event_id += 1
                affect_log.append(state['emotion']['affect'])
                phi_log.append(state['emotion']['phi'])
            print("10 steps run.")
        elif cmd == 'introspect':
            net.introspect()
            net.print_log(2)
        elif cmd.startswith('query '):
            layer = cmd.split(' ', 1)[1]
            print(net.query_layer(layer))
        elif cmd == 'save':
            net.save_state()
            print("State saved.")
        elif cmd == 'load':
            net.load_state()
            print("State loaded.")
        elif cmd == 'tags':
            tags = net.tag_important_events()
            print(f"Important tags: {tags}")
        elif cmd.startswith('get '):
            tag = cmd.split(' ', 1)[1]
            event = net.retrieve_event(tag)
            print(f"Event for {tag}: {event}")
        elif cmd == 'crisis':
            net.sensory.rewrite()
            net.perception.rewrite()
            net.narrative.rewrite()
            net.meta.rewrite()
            net.emotion.rewrite()
            net.self_model.rewrite()
            print("Identity crisis: all layers rewritten.")
        elif cmd == 'overload':
            for _ in range(30):
                x = generate_input(event_id, DEFAULT_CONFIG)
                net.forward(x)
                event_id += 1
            net.narrative.prune()
            net.self_model.prune()
            print("Memory overload: forced pruning.")
        elif cmd == 'phi':
            visualize_phi(phi_log)
        elif cmd == 'affect':
            visualize_affect(affect_log)
        elif cmd.startswith('multi '):
            n = int(cmd.split(' ', 1)[1])
            multi_agents = MultiAgentSystem(n, DEFAULT_CONFIG)
            print(f"Initialized {n} agents.")
        elif cmd == 'multi step' and multi_agents:
            multi_agents.step_all()
            print("All agents stepped.")
        elif cmd == 'multi introspect' and multi_agents:
            multi_agents.introspect_all()
        elif cmd.startswith('ckpt '):
            name = cmd.split(' ', 1)[1] if ' ' in cmd else None
            net.save_checkpoint(name)
        elif cmd == 'ckpt list':
            print(net.list_checkpoints())
        elif cmd.startswith('ckpt load '):
            name = cmd.split(' ', 2)[2] if len(cmd.split(' ', 2)) > 2 else None
            if name:
                net.load_checkpoint(name)
        elif cmd.startswith('replay '):
            parts = cmd.split()
            start = int(parts[1]) if len(parts) > 1 else 0
            end = int(parts[2]) if len(parts) > 2 else None
            net.replay_events(start, end)
        elif cmd.startswith('event '):
            idx = int(cmd.split(' ', 1)[1])
            net.print_event(idx)
        elif cmd == 'clearlog':
            net.clear_log()
        elif cmd == 'exit':
            print("Exiting.")
            break
        else:
            print("Unknown command. Use 'run', 'introspect', 'save', 'load', 'query [layer]', 'tags', 'get [tag]', 'crisis', 'overload', 'phi', 'affect', 'multi [n]', 'multi step', 'multi introspect', 'ckpt [name]', 'ckpt list', 'ckpt load [name]', 'replay [start] [end]', 'event [idx]', 'clearlog', or 'exit'.")

"""
USAGE HINTS:
- Use 'ckpt [name]' to save a checkpoint, 'ckpt list' to list, and 'ckpt load [name]' to restore.
- Use 'replay [start] [end]' to replay events, and 'event [idx]' to inspect a specific event.
- Use 'clearlog' to clear all logs.
- All features are scriptable via the ConsciousNet API for automation and experiments.
"""
# Test/demo code
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        run_demo(steps=50, introspect_every=10)
    else:
        main()
