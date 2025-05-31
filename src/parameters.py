params = {
    # dataset 관련 파라미터
    
    # 데이터 경로
    "INPUT_LEFT_DIR":        "dataset/clipped_left",  # 원본 WAV 파일이 들어 있는 디렉토리
    "CACHE_LEFT_DIR":        "dataset/cache/left",    # 로그 멜 스펙토그램 캐시(.npy)를 저장할 디렉토리

    "INPUT_RIGHT_DIR":        "dataset/clipped_right",  # 원본 WAV 파일이 들어 있는 디렉토리
    "CACHE_RIGHT_DIR":        "dataset/cache/right",    # 로그 멜 스펙토그램 캐시(.npy)를 저장할 디렉토리

    "INPUT_WHITE_NOISE_DIR":        "dataset/clipped_white_noise",  # 원본 WAV 파일이 들어 있는 디렉토리
    "CACHE_WHITE_NOISE_DIR":        "dataset/cache/white_noise",    # 로그 멜 스펙토그램 캐시(.npy)를 저장할 디렉토리
    
    "LOG_DIR":          "logs",  # 로그 파일을 저장할 디렉토리
    "CHECKPOINTS_DIR":  "checkpoints",  # 모델 체크포인트를 저장할 디렉토리
    "OUTPUT_DIR":       "outputs",  # 모델 예측 결과를 저장할 디렉토리
    
    # 오디오 전처리 파라미터
    "SAMPLE_RATE":      24000,   # librosa.load 시 사용할 샘플링 레이트 (Hz)
    "N_FFT":            512,     # STFT 계산 시 FFT 윈도우 크기 (샘플 수)
    "HOP_LENGTH":       256,     # STFT 계산 시 인접 프레임 간 샘플 오프셋
    "WIN_LENGTH":       512,     # STFT 윈도우 길이 (샘플 수)

    # 멜 스펙트로그램 파라미터
    "nb_mels":          64,       # 멜 필터 뱅크 개수
    
    "BATCH_SIZE":      32,      # 데이터 로더 배치 크기
    
    "TRAIN_RATIO":     0.8,     # 전체 데이터 중 훈련용 비율 (나머지는 검증용)
    
    "input_channels": 2,  # 입력 오디오 채널 수 (예: 스테레오 = 2, 모노 = 1)
    
    # CNN Frontend
    'nb_conv_filters': 128,
    'f_pool_size': [2, 2, 2],
    'dropout': 0.05,

    # Patch Embedding
    'patch_size': 7,
    'stride_size': 3,
    'embed_dim': 128,
    'patch_padding': 3,

    # TCN Block
    'tcn_dilation_sets': [[1, 2, 4], [2, 4, 8], [1, 2, 4], [2, 4, 8]],

    # Global Pooling
    'pooling_length': 50,

    # Output Head
    'fc_dims': [64],

    # Training
    'nb_epochs': 200,
    'batch_size': 150,
    'nb_workers': 4,
    'shuffle': True,

    # Optimizer
    'learning_rate': 0.0010997,
    'weight_decay': 7.817e-05,

    # 기타 저장용
    'model_dir': 'tcn_classwise_checkpoints'  # best_model.pth 저장 경로
}