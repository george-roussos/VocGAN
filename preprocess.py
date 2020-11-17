import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from utils.stft import TacotronSTFT
from utils.hparams import HParam
from utils.utils import read_wav_np
from mozilla_tts_audio import AudioProcessor


def main(hp, args):
    ap = AudioProcessor(                 
       sample_rate=22050,
       num_mels=80,
       min_level_db=-100,
       frame_shift_ms=None,
       frame_length_ms=None,
       hop_length=256,
       win_length=1024,
       ref_level_db=20,
       fft_size=1024,
       power=1.5,
       preemphasis=0.98,
       signal_norm=True,
       symmetric_norm=True,
       max_norm=4.0,
       mel_fmin=0.0,
       mel_fmax=8000.0,
       spec_gain=20.0,
       stft_pad_mode="reflect",
       clip_norm=True,
       griffin_lim_iters=60,
       do_trim_silence=False,
       trim_db=60)


    wav_files = glob.glob(os.path.join(args.data_path, '**', '*.wav'), recursive=True)
    mel_path = hp.data.mel_path

    # Create all folders
    os.makedirs(mel_path, exist_ok=True)
    for wavpath in tqdm.tqdm(wav_files, desc='preprocess wav to mel'):
        sr, wav = read_wav_np(wavpath)
        assert sr == hp.audio.sampling_rate, \
            "sample rate mismatch. expected %d, got %d at %s" % \
            (hp.audio.sampling_rate, sr, wavpath)
        
        if len(wav) < hp.audio.segment_length + hp.audio.pad_short:
            wav = np.pad(wav, (0, hp.audio.segment_length + hp.audio.pad_short - len(wav)), \
                    mode='constant', constant_values=0.0)

        wav = torch.from_numpy(wav).unsqueeze(0)
        wav = wav.squeeze(0)
        mel = np.float32(ap.melspectrogram(wav.detach().cpu().numpy()))
        mel = torch.from_numpy(mel)
        mel = mel.unsqueeze(0)
        mel = mel.squeeze(0)  # [num_mel, T]
        id = os.path.basename(wavpath).split(".")[0]
        np.save('{}/{}.npy'.format(mel_path, id), mel.numpy(), allow_pickle=False)
        #torch.save(mel, melpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for config.")
    parser.add_argument('-d', '--data_path', type=str, required=True,
                        help="root directory of wav files")
    args = parser.parse_args()
    hp = HParam(args.config)

    main(hp, args)
