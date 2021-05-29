import json
import struct
from pathlib import Path

import scipy as sp
from librosa.core import audio
from wavinfo import WavInfoReader


SR = 16000
CHANNELS = 1
CODEC = "pcm_s16le"
sample_rate = 16000
window_time = 0.020 # seconds
window_size = int(sample_rate * window_time)
spectogram = audio.transforms.Spectrogram(
    n_fft=window_size,
    win_length=window_size,
    hop_length=window_size,
    power=1
)
FFPROBE_CMD = (
    "ffprobe -v quiet -print_format json "
    "-show_entries stream=duration,sample_rate,codec_name,channels {file}"
)
AUTOFIX_CMD = (
    f"ffmpeg -hide_banner -loglevel error -i {{input}} "
    f"-c:a {CODEC} -ar {SR} -ac {CHANNELS} -y {{output}}"
)

def validate_wav(path, autofix=False):
    cmd = FFPROBE_CMD.format(file=path)
    out = sp.check_output(cmd, shell=True, text=True)
    data = json.loads(out)["streams"][0]

    good = (
            data["codec_name"] == CODEC
            and int(data["sample_rate"]) == SR
            and data["channels"] == CHANNELS
    )

    if not good and autofix:
        output = path.parent / "autofix" / Path(path).name
        print(f"Autofixing {path} -> {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = AUTOFIX_CMD.format(input=path, output=output)
        sp.check_output(cmd, shell=True, text=True)
        data['autofix_path'] = output

    return good, data


def extract_wav_regions(path):
    with open(path, 'rb') as file:
        content = file.read()

    try:
        info = WavInfoReader(path)
        parent_chunks_dict = {x[0].decode("latin-1"): x for x in info.main_list}
    except Exception as err:
        # print('WavInfo error:', err)
        return None

    ok = "cue " in parent_chunks_dict and "adtl" in parent_chunks_dict
    if not ok:
        # print('Missing chunks (CUE, ADTL) in', path)
        return None

    offset = parent_chunks_dict["cue "].start
    length = struct.unpack_from('I', content, offset)[0]

    offset += 4
    cue_points = {}
    for i in range(length):
        id, pos, *_ = struct.unpack_from('IIIIII', content, offset)
        cue_points[id] = pos
        offset += 24

    region_chunks = parent_chunks_dict["adtl"].children
    regions = []
    for chunk in region_chunks:
        id, length, *_ = struct.unpack_from('IIIHHHH', content, chunk.start)
        regions.append((cue_points[id], cue_points[id] + length))

    return regions
