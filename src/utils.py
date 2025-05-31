import numpy as np
import librosa
import torch

def extract_stft(audio, n_fft, hop_length, win_length):
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T
    return stft

def extract_log_mel_spectrogram(audio, sr, n_fft, hop_length, win_length, nb_mels):
    """
    오디오 신호로부터 로그 멜 스펙트로그램을 계산합니다.

    파라미터:
        audio (ndarray): 오디오 파형을 담고 있는 NumPy 배열 (1D 또는 다채널). 
                          일반적으로 float32 형태의 샘플 값들을 포함합니다.
        sr (int): 오디오 신호의 샘플링 레이트 (예: 24000, 48000 등).
        n_fft (int): FFT 윈도우 길이 (샘플 수, 예: 512, 1024).
        hop_length (int): 인접 프레임 사이의 샘플 간격 (얼마만큼 옆으로 이동하며 FFT를 계산할지).
        win_length (int): 각 윈도우(프레임)를 생성할 때 사용하는 실제 창의 길이 (샘플 수).
        nb_mels (int): 멜 필터뱅크의 개수 (예: 40, 64).

    반환값:
        ndarray: 형태가 (채널, 시간 프레임, nb_mels)인 로그 멜 스펙트로그램 배열.
                 - 채널 차원이 2라면 (2, time_frames, nb_mels).
                 - 단일 채널이라면 (1, time_frames, nb_mels) 혹은 (time_frames, nb_mels)로 해석 가능합니다.

    동작 과정:
        1. extract_stft 함수를 호출하여 입력 오디오의 복소수 STFT를 계산한다.
        2. 복소수 STFT의 절댓값을 제곱하여 파워 스펙트럼을 얻는다.
        3. librosa.feature.melspectrogram을 사용해 파워 스펙트럼으로부터 멜 스펙트로그램(mel_spec)을 계산한다.
        4. librosa.power_to_db를 적용해 멜 스펙트로그램을 데시벨(dB) 스케일로 변환한다.
        5. 최종적으로 (nb_mels, time_frames) 형태인 로그 멜 스펙트로그램을 
           transpose((2, 0, 1))를 통해 (채널, time_frames, nb_mels) 형태로 바꿔 반환한다.
    """
    linear_stft = extract_stft(audio, n_fft, hop_length, win_length)
    linear_stft_mag = np.abs(linear_stft) ** 2
    mel_spec = librosa.feature.melspectrogram(S=linear_stft_mag, sr=sr, n_mels=nb_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spec)
    log_mel_spectrogram = log_mel_spectrogram.transpose((2, 0, 1))
    return log_mel_spectrogram
    
def decode_logits(logits: torch.Tensor,
                          threshold: float = 0.5
                         ) -> list[int]:
    """
    logits: 모델이 반환한 텐서. 다음 두 가지 형태 모두 지원합니다:
        - [1, 2] 형태 (batch_size=1로 반환된 경우)
        - [2] 형태 (이미 배치 차원을 제거한 경우)
    threshold: 확률을 0/1로 판단할 기준값 (기본 0.5)

    반환값: [is_left_int, is_right_int]  (각각 0 또는 1)
    """
    # 1) logits 차원 정리
    if logits.ndim == 2 and logits.size(0) == 1 and logits.size(1) == 2:
        probs = logits.squeeze(0)   # [1, 2] -> [2]
    elif logits.ndim == 1 and logits.size(0) == 2:
        probs = logits
    else:
        raise ValueError(f"지원되지 않는 logits 크기입니다: {logits.shape}. \
함수는 [1,2] 또는 [2] 크기의 텐서만 처리합니다.")

    # 2) CPU로 이동시키고 float 값으로 변환
    left_prob  = float(probs[0].cpu().item())
    right_prob = float(probs[1].cpu().item())

    # 3) threshold 기준으로 0 또는 1 판단
    is_left_int  = 1 if left_prob  > threshold else 0
    is_right_int = 1 if right_prob > threshold else 0

    return [is_left_int, is_right_int]

