from pathlib import Path
import numpy as np
import librosa
import json
import sys

ROOT = Path(__file__).parent
SAMPLES_PATH = ROOT / 'samples'
META_PATH = SAMPLES_PATH / 'meta.json'
META = json.load(open(META_PATH))

print(
    'ALINA SAMPLES:',
    ', '.join((f'{k.upper()}: {len(v)}' for k, v in META.items())),
    file=sys.stderr,
)


def get_alina_sample(src, sr=16000):
    item = np.random.choice(META[src])
    path = SAMPLES_PATH / item['path']
    sig, _ = librosa.load(str(path), sr=sr)
    meta = {'src': src, 'path': str(path)}
    if item.get('regions') is not None:
        index = np.random.choice(np.arange(0, len(item['regions'])))
        meta['start'], meta['end'] = item['regions'][index]
    return sig, meta
