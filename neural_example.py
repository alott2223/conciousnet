import torch
import matplotlib.pyplot as plt
from conscious_net import ConsciousNet, DEFAULT_CONFIG, preprocess_vision, preprocess_text, preprocess_abstract

NUM_EVENTS = 20
VISION_DIM = DEFAULT_CONFIG['sensory']['vision_dim']
TEXT_VOCAB = DEFAULT_CONFIG['sensory']['text_vocab']
ABSTRACT_DIM = DEFAULT_CONFIG['sensory']['abstract_dim']

net = ConsciousNet(DEFAULT_CONFIG)
phi_log = []
affect_log = []

for i in range(NUM_EVENTS):
    vision = preprocess_vision(None, input_dim=VISION_DIM)
    text = preprocess_text(None, vocab_size=TEXT_VOCAB, seq_len=10)
    abstract = preprocess_abstract(None, input_dim=ABSTRACT_DIM)
    x = {'vision': vision, 'text': text, 'abstract': abstract}
    state = net.forward(x)
    net.feedback_recursion()
    net.adaptive_restructuring()
    phi_log.append(state['emotion']['phi'])
    affect_log.append(state['emotion']['affect'])

net.introspect()
net.print_log(3)
net.save_checkpoint('neural_example_checkpoint')

print('\n--- Replay first 5 events ---')
net.replay_events(0, 5)

plt.figure(figsize=(8, 3))
plt.plot(phi_log, label='Phi (Significance)')
plt.xlabel('Event')
plt.ylabel('Phi')
plt.title('Phi Evolution Over Time')
plt.legend()
plt.tight_layout()
plt.show()

affect_arr = torch.tensor(affect_log)
plt.figure(figsize=(8, 3))
for i in range(affect_arr.shape[1]):
    plt.plot(affect_arr[:, i], label=f'Affect Dim {i}')
plt.xlabel('Event')
plt.ylabel('Affect Value')
plt.title('Affective State Evolution')
plt.legend()
plt.tight_layout()
plt.show()
