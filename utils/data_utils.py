import os
import jsonlines
import subprocess
import numpy as np
import torch.nn
import torchaudio
from utils.basic_utils import check_dirs
from torchvision import transforms
from decord import VideoReader
from decord import cpu, gpu


def load_metas(meta_file):
    metas = []
    with jsonlines.open(meta_file, mode='r') as rfile:
        for line in rfile:
            metas.append(line)
    return metas


def video_duration(timestamp1, timestamp2):
    hh, mm, s = timestamp1.split(':')
    ss, ms = s.split('.')
    timems1 = 3600 * 1000 * int((hh)) + 60 * 1000 * int(mm) + 1000 * int(ss) + int(ms)
    hh, mm, s = timestamp2.split(':')
    ss, ms = s.split('.')
    timems2 = 3600 * 1000 * int((hh)) + 60 * 1000 * int(mm) + 1000 * int(ss) + int(ms)
    dur = (timems2 - timems1) / 1000
    return str(dur)


def vision_clip_extract(input_file, output_file, start, end):
    vision_cmd = ['ffmpeg',
                  '-ss', start,
                  '-t', video_duration(start, end),
                  '-accurate_seek',
                  '-i', input_file,
                  '-c', 'copy',
                  '-avoid_negative_ts', '1',
                  '-reset_timestamps', '1',
                  '-y',
                  '-hide_banner',
                  '-loglevel', 'quiet',
                  '-map', '0',
                  output_file]
    res = subprocess.run(vision_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    return res


def audio_extract(input_file, output_file):
    audio_cmd = ['ffmpeg',
                 '-i', input_file,
                 '-acodec', 'pcm_s16le',
                 '-vn',
                 '-ac', '1',
                 '-ar', '16000',
                 '-f', 'wav',
                 '-loglevel', 'quiet',
                 output_file]
    res = subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    return res


def _preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])


def load_video(file, target_frames):
    try:
        vr = VideoReader(file, ctx=cpu(0))
        total_frame_num = len(vr)
        sample_rate = max(total_frame_num // target_frames, 1)

        frame_idx = list(range(0, min(total_frame_num, sample_rate * target_frames), sample_rate))
        img_array = vr.get_batch(frame_idx).asnumpy()  # (num_frames, H, W, 3)

        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2)
        return img_array    # [target_frames, 3, H, W]
    except:
        print(f"Load Video Error {file}")
        return None


def load_wav(wav_file, target_frames):
    waveform, sample_rate = torchaudio.load(wav_file, normalize=True)
    total_frames = waveform.shape[-1]
    pad_frames = target_frames - total_frames

    if pad_frames > 0:
        waveform = torch.nn.ZeroPad2d(padding=(0, pad_frames, 0, 0))(waveform)
    elif pad_frames < 0:
        waveform = waveform[:, :target_frames]

    return waveform     # [1, target_frames]


def wav2fbank(wav_file, target_frames):
    waveform, sample_rate = torchaudio.load(wav_file, normalize=True)
    fbank = torchaudio.compliance.kaldi.fbank(waveform,
                                              htk_compat=True,
                                              sample_frequency=sample_rate,
                                              window_type="hanning",
                                              num_mel_bins=128)

    total_frames = fbank.shape[0]
    pad_frames = target_frames - total_frames

    if pad_frames > 0:
        fbank = torch.nn.ZeroPad2d(padding=(0, 0, 0, pad_frames))(fbank)
    elif pad_frames < 0:
        fbank = fbank[:target_frames, :]

    return fbank    # [frames, freq]


def vision_frames_extract(input_file, output_path, fps=0.5):
    base_name = os.path.basename(input_file).split(".")[0]
    check_dirs(output_path)

    vision_cmd = ['ffmpeg',
                  '-y',
                  '-loglevel', 'quiet',
                  '-i', input_file,
                  '-vf', f'fps={fps}',
                  '-q:v', '2',
                  '-f', 'image2',
                  f'{output_path}/{base_name}.%06d.jpg']
    res = subprocess.run(vision_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    return res


def vision_frames_clip_extract(input_file, output_path, start, end, fps=1):
    check_dirs(output_path)
    base_name = output_path.split('/')[-1]
    vision_cmd = ['ffmpeg',
                  '-y',
                  '-ss', start,
                  '-t', video_duration(start, end),
                  '-accurate_seek',
                  '-i', input_file,
                  '-avoid_negative_ts', '1',
                  '-vf', f'fps={fps}',
                  '-q:v', '2',
                  '-f', 'image2',
                  '-loglevel', 'quiet',
                  f'{output_path}/{base_name}_%06d.jpg']
    res = subprocess.run(vision_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    return res


def audio_clip_extract(input_file, output_file, start, end):
    audio_cmd = ['ffmpeg',
                 '-ss', start,
                 '-t', video_duration(start, end),
                 '-accurate_seek',
                 '-i', input_file,
                 '-vn',
                 '-acodec', 'pcm_s16le',
                 '-ac', '1',
                 '-ar', '16000',
                 '-f', 'wav',
                 '-loglevel', 'quiet',
                 output_file]
    res = subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
    return res


def _zeroOneNormalize(image):
    return image.float().div(255)


def image_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(n_px),
        _zeroOneNormalize,
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

