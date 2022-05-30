# Modified by nemoramo.
# This script slightly change the original code from NeMo. The split mechanism
# is based on subsegments and uses torchaudio to speed up duration calculation.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import random
import torchaudio
# multi processing executor
from loky import get_reusable_executor

from tqdm import tqdm
from pydub import AudioSegment

random.seed(42)

"""
This script converts a scp file where each line contains  
<absolute path of wav file> 
to a manifest json file. 
Args: 
--scp: scp file name
--id: index of speaker label in filename present in scp file that is separated by '/'
--out: output manifest file name
--split: True / False if you would want to split the  manifest file for training purposes
        you may not need this for test set. output file names is <out>_<train/dev>.json
        Defaults to False
--num_test: the number of segments per speaker in valid part
--create_chunks: bool if you would want to chunk each manifest line to chunks of 3 sec or less
        you may not need this for test set, Defaults to False
--min_durations: random choice of segment durations
"""


def get_file_duration(path):
    """
    Returns the duration of a wav file in seconds.
    :param path: audio file path
    :return: duration of the audio file in seconds
    """
    try:
        meta = torchaudio.info(path)
        wav_len = round(meta.num_frames / meta.sample_rate, 5)
        sr = meta.sample_rate
    except RuntimeError:
        # unrecognized audio formats
        sample = AudioSegment.from_file(path)
        wav_len = len(sample) / 1000
        sr = sample.frame_rate
    return wav_len, sr


def filter_manifest_line(manifest_line):
    """Filters a manifest line to chunks of seconds in min_durations.
    :param manifest_line: manifest line to filter
    :return: chunks of manifest line
    """
    split_manifest = []
    speakers = []
    audio_path = manifest_line['audio_filepath']
    start = manifest_line.get('offset', 0)
    dur = manifest_line['duration']
    SPKR = manifest_line['label']

    if dur >= 0.5:
        remaining_dur = dur
        temp_dur = random.choice(MIN_DURATIONS)
        remaining_dur = remaining_dur - temp_dur
        while remaining_dur > 0:
            meta = {'audio_filepath': audio_path, 'offset': start, 'duration': temp_dur, 'label': SPKR}
            split_manifest.append(meta)
            speakers.append(SPKR)

            start = start + temp_dur
            temp_dur = random.choice(MIN_DURATIONS)
            remaining_dur = remaining_dur - temp_dur

    return split_manifest, speakers


def write_file(name, lines, idx):
    with open(name, 'w') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')
    logging.info("wrote", name)


def worker(scp_line, idx=0, create_chunks=False):
    """
    Worker function for multiprocessing scp line
    :param scp_line: line in scp file, format: <wav_id> <absolute path of wav file>
    :param idx: speaker placement index
    :return: sub meta lines, sub speakers, sub spk2utts
    """
    scp_line = scp_line.strip()
    utt_id, wav_path = scp_line.split(' ')
    dur, sr = get_file_duration(wav_path)
    speaker = utt_id.split('-')[idx]
    speaker = list(speaker)
    speaker = ''.join(speaker)
    meta = [{'audio_filepath': wav_path, 'duration': dur, 'label': speaker, 'offset': 0}]
    speakers = [speaker]
    if create_chunks:
        meta, speakers = filter_manifest_line(meta[0])
    spk2utts = {speaker: meta}
    return meta, speakers, spk2utts


def main(scp, id, out, split=False, create_chunks=False, nj=1, debug_num=-1):
    if os.path.exists(out):
        os.remove(out)
    scp_file = open(scp, 'r').readlines()
    scp_file = sorted(scp_file)
    scp_file = scp_file[:debug_num] if debug_num > 0 else scp_file

    executor = get_reusable_executor(max_workers=nj)
    results = executor.map(worker, scp_file, [id] * len(scp_file), [create_chunks] * len(scp_file))
    submetas, subspeakers, subspk2utts = zip(*results)

    lines = []
    speakers = []
    spk2utts = {}
    for submeta, subspk, subspk2utt in zip(submetas, subspeakers, subspk2utts):
        lines.extend(submeta)
        speakers.extend(subspk)
        for key in subspk2utt:
            if key in spk2utts:
                spk2utts[key].extend(subspk2utt[key])
            else:
                spk2utts[key] = subspk2utt[key]

    path = os.path.dirname(out)
    os.makedirs(path, exist_ok=True)
    write_file(out, lines, range(len(lines)))
    if split:
        train_lines = []
        valid_lines = []
        for speaker, utts in spk2utts.items():
            if len(utts) > NUM_TEST:
                random.shuffle(utts)
                for idx, utt in enumerate(utts):
                    if idx < NUM_TEST:
                        valid_lines.append(utt)
                    else:
                        train_lines.append(utt)
            else:
                train_lines.extend(utts)

        out = os.path.join(path, 'train.manifest')
        write_file(out, train_lines, range(len(train_lines)))
        out = os.path.join(path, 'dev.manifest')
        write_file(out, valid_lines, range(len(valid_lines)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scp", help="scp file name", type=str, required=True)
    parser.add_argument(
        "--id", help="field num seperated by '-' to be considered as speaker label", type=int, required=True
    )
    parser.add_argument("--out", help="manifest_file name", type=str, required=True)
    parser.add_argument(
        "--split",
        help="bool if you would want to split the manifest file for training purposes",
        required=False,
        action='store_true',
    )
    parser.add_argument(
        "--num_test",
        help="the number of segments per speaker in valid part",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--create_chunks",
        help="bool if you would want to chunk each manifest line to chunks of 3 sec or less",
        required=False,
        action='store_true',
    )
    parser.add_argument(
        "--min_durations",
        nargs="+",
        type=float,
        default=[1.5, 2]
    )
    parser.add_argument(
        "--nj",
        help="number of processes",
        default=1,
        type=int
    )
    parser.add_argument(
        "--debug_num",
        help="number of lines to process",
        default=-1,
        type=int
    )
    args = parser.parse_args()

    MIN_DURATIONS = args.min_durations
    NUM_TEST = args.num_test
    main(args.scp, args.id, args.out, args.split, args.create_chunks, args.nj, args.debug_num)