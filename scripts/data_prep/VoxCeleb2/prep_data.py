import argparse
import glob
import hashlib
import os.path
import subprocess
import zipfile

from omegaconf import OmegaConf
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser("VoxCeleb2 dataset preparation")
    parser.add_argument("--save_path", required=True, type=str,
                        help="where to save downloaded data")
    parser.add_argument("--extract_path", required=True, type=str,
                        help="where to extract data")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--ext", type=str, default=".m4a")
    parser.add_argument("--scp_out", type=str, required=True)
    parser.add_argument("--download", action="store_true",
                        help="download data button")
    parser.add_argument("--concat", action="store_true",
                        help="whether to concatenate audios in one series")
    parser.add_argument("--convert", action="store_true",
                        help="whether to convert m4a format file to wav")
    args = parser.parse_args()
    return args


def md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(download_list, md5_list, args):
    for idx, url in enumerate(download_list):
        outfile = url.split('/')[-1]
        sub_save_path = os.path.join(args.save_path, outfile)
        out = subprocess.call(
            'wget %s -O %s' % (url, sub_save_path),
            shell=True)
        if out != 0:
            raise ValueError(
                'Download failed %s.' % url)
        md5check = md5(sub_save_path)
        assert md5check == md5_list[idx], "{} download failed due to md5 mismatch".format(outfile)


def concatenate_unzip(save_path, extract_path):
    _ = subprocess.call('cat %s/vox2_dev_aac* > %s/vox2_aac.zip' % save_path, shell=True)
    zip_file = os.path.join(save_path, "vox2_aac.zip")
    unzip(zip_file, extract_path)


def unzip(zip_file, extract_path):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def convert(extract_path, sample_rate):
    files = glob.glob('{}/dev/aac/*/*/*.m4a'.format(extract_path))
    print("Total files contained in {} : {}".format(extract_path, len(files)))

    for fname in tqdm(files):
        outfile = fname.replace('.m4a', '.wav')
        out = subprocess.call(
            'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le '
            '-ar %d %s >/dev/null 2>/dev/null' % (fname, sample_rate, outfile),
            shell=True)
        if out != 0:
            raise ValueError('Conversion failed %s.' % fname)


def generate_scp(wav_path, suffix='.wav'):
    files = glob.glob('{}/dev/aac/*/*/*{}'.format(wav_path, suffix))
    scps = []
    for fpath in tqdm(files):
        fname = "-".join(fpath.split("/")[-3:])[:-len(suffix)]
        scps.append([fname, fpath])
    return scps


def main():
    args = _parse_args()
    if args.download:
        yaml = OmegaConf.load("download_meta.yaml")
        download_list = [item_pair[0] for item_pair in yaml.train]
        md5_list = [item_pair[1] for item_pair in yaml.train]
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.extract_path)
        download(download_list, md5_list, args)
        concatenate_unzip(args.save_path, args.extract_path)

    # For speaker verification task, audio concatenation also works.
    if args.concat:
        pass
    
    # This repo also supports m4a format using pydub which means this process may not be necessary.
    if args.convert:
        print('Converting files from AAC to WAV')
        convert(args.extract_path, args.sample_rate)

    scps = generate_scp(args.extract_path, suffix=args.ext)
    out_dir = os.path.dirname(args.scp_out)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.scp_out, 'w') as f:
        for scp in scps:
            f.write(" ".join(scp) + "\n")


if __name__ == '__main__':
    main()