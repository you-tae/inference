"""
parameters.py

This module stores all the configurable parameters and hyperparameters used
across the project, ensuring easy tuning and reproducibility.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: January 2025
"""

params = {

    # choose task
    'modality': 'audio',  # 'audio' or 'audio_visual'
    'net_type': 'SELDnet',

    # data params
    'root_dir': '/home/kwak/SELD_task/dataset',  # parent directory containing the audio, video and labels directory
    'feat_dir': '/home/kwak/SELD_task/dataset/features',  # store extracted features here
    

    'log_dir': 'logs',  # save all logs here like loss and metrics
    'checkpoints_dir': 'checkpoints',  # save trained model checkpoints and config
    'output_dir': 'outputs',  # save the predicted files here.


    # audio feature extraction params
    'sampling_rate': 24000,
    'hop_length_s': 0.02,
    'nb_mels': 64,
    'input_channels': 2,           # 오디오 채널 수

    # video feature extraction params
    'fps': 10,
    'resnet_feature_size': 49,  # (7,7) feature_map for every frame from resnet

    # ▶ CNN Frontend
    'nb_conv_blocks': 3,
    'nb_conv_filters':128,         # Conv2D 채널 수
    'f_pool_size': [2, 2, 2],         # 주파수 방향 MaxPool 비율
    't_pool_size': [5, 1, 1],      # (기존 SELDNet용, 사용 안 해도 무방)
    'dropout': 0.05,                # Dropout 비율

    # ▶ Patch Embedding
    'patch_size': 3,               # Conv1D patch 크기
    'stride_size': 2,              # overlap 정도
    'patch_padding': 0,            # patch padding (보통 patch_size // 2)
    'embed_dim': 128,              # patch embedding 출력 차원

    # ▶ TCN Block
    'tcn_dilation_sets': [[1, 2, 4], [2, 4, 8], [1, 2, 4], [2, 4, 8]],

    'rnn_size': 128,
    'nb_rnn_layers': 2,

    'nb_self_attn_layers': 2,
    'nb_attn_heads': 8,

    'nb_transformer_layers': 2,

    'nb_fnn_layers': 1,
    'fnn_size':128,

    # ▶ Global Pooling
    'pooling_length': 50,          # adaptive pooling size (T dimension)

    # ▶ Output Head (FC layers)
    'fc_dims': [256, 128],         # FC 레이어 차원

    # ▶ Output
    'nb_classes': 13,              # 클래스 수
    'max_polyphony': 3,            # 동시 등장하는 소스 수 (Multi-ACCDOA)

    'label_sequence_length': 50,  # 5 seconds with 100ms frames

    # loss params
    'multiACCDOA': True,
    'thresh_unify': 15,

    # training params
    'nb_epochs': 200,
    'batch_size': 150,
    'nb_workers': 12,
    'shuffle': True,

    # optimizer params
    'learning_rate': 0.0010997,
    'weight_decay': 7.817e-05,

    # folds for training, testing
    'dev_train_folds': ['fold1', 'fold3'],  # 'fold1' is the synthetic training data. You can skip that if you do not use the synthetic data to train.
    'dev_test_folds': ['fold4'],

    # metric params
    'average': 'macro',                  # Supports 'micro': sample-wise average and 'macro': class-wise average.
    'segment_based_metrics': False,      # If True, uses segment-based metrics, else uses event-based metrics.
    'lad_doa_thresh': 20,                # DOA error threshold for computing the detection metrics.
    'lad_dist_thresh': float('inf'),     # Absolute distance error threshold for computing the detection metrics.
    'lad_reldist_thresh': float('1.0'),  # Relative distance error threshold for computing the detection metrics.
    'lad_req_onscreen': False,           # Require correct on-screen estimation when computing the detection metrics.

    'use_jackknife': False,               # If True, uses jackknife to calc results of the best model on test/eval set.
                                          # CAUTION: Too slow to use jackknife

}
