import torch
import torchaudio
import torchcrepe

_meltransformer = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=80,
    f_min=80.0,
    f_max=7600.0,
    power=1.0,  # magnitude mel, then take log
    mel_scale="slaney",
    norm="slaney",
)


def to_logmelspec(waveform, sr):
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    _meltransformer.to(device=waveform.device)
    mel = _meltransformer(waveform)
    logmel = torch.log10(torch.clamp(mel, min=1e-5)).transpose(-1, -2)
    return logmel


def display_audio(waveform, sr):
    import IPython.display as ipd

    ipd.display(ipd.Audio(waveform, rate=sr))


def extract_f0_torchcrepe(wav_16k: torch.Tensor, sr=16000, hop=256):
    f0 = torchcrepe.predict(
        wav_16k,
        sr,
        hop,
        fmin=50.0,
        fmax=1100.0,
        model="full",
        device=wav_16k.device,  # type: ignore
        batch_size=1024,
    )
    # Convert Hz→MIDI (or z-score) for smoother learning
    f0_midi = 69 + 12 * torch.log2(torch.clamp(f0, min=1e-6) / 440.0)  # type: ignore
    f0_midi[~torch.isfinite(f0_midi)] = 0.0
    f0_midi = torch.nan_to_num(f0_midi, nan=0.0, posinf=0.0, neginf=0.0)
    return f0_midi  # (T)
