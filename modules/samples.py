import json
import re
import sys
from pathlib import Path
from pprint import pprint

import librosa
import numpy as np
import torchaudio as audio

from modules.audio import validate_wav, extract_wav_regions

SAMPLES_PATH = Path('/content/drive/MyDrive/alina_clean-2/samples')

FILE_PATTERNS = {
    'pos_clean': 'pos_clean_{}.wav',
    'pos_noisy': 'pos_noisy_{}.wav',
    'neg_clean': 'neg_clean_{}.wav',
    'neg_noisy': 'neg_noisy_{}.wav',
    'neg_random': 'neg_random_{}.wav',
}
all_labels = {key: [] for key in FILE_PATTERNS}

ROOT = Path('/content/drive/MyDrive/alina_clean-2').parent
SAMPLES_PATH = ROOT / 'alina_clean-2/samples'
META_PATH = SAMPLES_PATH / 'meta.json'
META = json.load(open(META_PATH))




def load_samples(filepath):
    channels, sample_rate = audio.load(filepath, channels_first=True, normalize=True)
    return channels[0]


def get_alina_sample(src, sr=16000):
    item = np.random.choice(META[src])
    path = SAMPLES_PATH / item['path']
    sig, _ = librosa.load(str(path), sr=sr)
    meta = {'src': src, 'path': str(path)}
    if item.get('regions') is not None:
        index = np.random.choice(np.arange(0, len(item['regions'])))
        meta['start'], meta['end'] = item['regions'][index]
    return sig, meta


def load_samples_from_files():
    bad = total = 0
    for dir in SAMPLES_PATH.glob('*'):
        if not str(dir.name).isnumeric() or dir.is_file():
            continue
        print(f'Checking {dir}...')
        for file in dir.glob('*.wav'):
            good, data = validate_wav(file, autofix=True)
            total += 1
            if not good:
                print(f'Bad file: {file}')
                pprint(data, width=120, stream=sys.stderr)
                print(f'Autofix output:', data.get('autofix_path', None))
                bad += 1

    print()
    print('TOTAL SAMPLES:', total)
    print('GOOD SAMPLES:', total - bad)
    print('BAD SAMPLES:', bad)

    assert bad == 0

    all_labels = {key: [] for key in FILE_PATTERNS}

    for dir in SAMPLES_PATH.glob('*'):
        if not str(dir.name).isnumeric() or dir.is_file():
            continue
        print(f'\nProcessing {dir}...')
        files = list(dir.glob('*.wav'))
        for key, fmt in FILE_PATTERNS.items():
            pattern = re.compile(fmt.format('\d+'))
            filtered = [x for x in files if pattern.fullmatch(x.name)]
            if not filtered:
                print('Group not found:', key)
                continue

            regions = None
            if 'random' not in key:
                main_file = dir / fmt.format(0)
                regions = extract_wav_regions(main_file)
            if regions is None:
                if (dir / 'meta.json').exists():
                    with open(dir / 'meta.json') as file:
                        meta = json.load(file)
                        regions = meta[key]['labels']
                elif main_file.with_suffix('.npy').exists():
                    regions = np.load(main_file.with_suffix('.npy'))
                    regions = regions.tolist()
                else:
                    print('No regions found in', main_file)
                    continue

        for file in filtered:
            all_labels[key].append({
                'path': file.relative_to(SAMPLES_PATH).as_posix(),
                'regions': regions,
            })
        print(f'+ {len(filtered)} files -> {key}')
