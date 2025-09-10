# CAGE
Code for paper "Enhancing Video Captioning with Contextual Anchor Guided Semantic Modeling"
Usage
Our proposed HMN is implemented with PyTorch.

Environment
Python = 3.7
PyTorch = 1.4
1.Installation

Context features (2D CNN features) : MSRVTT-InceptionResNetV2
Motion features (3D CNN features) : MSRVTT-C3D
Object features (Extracted by Faster-RCNN) : MSRVTT-Faster-RCNN
Linguistic supervision: MSRVTT-Language
Splits: MSRVTT-Splits
MSVD Dataset:

Context features (2D CNN features) : MSVD-InceptionResNetV2
Motion features (3D CNN features) : MSVD-C3D
Object features (Extracted by Faster-RCNN) : MSVD-Faster-RCNN
Linguistic supervision: MSVD-Language
Splits: MSVD-Splits
3.Prepare training data
Organize visual and linguistic features under data/
data
├── __init__.py
├── loader
│   ├── data_loader.py
│   └── __init__.py
├── MSRVTT
│   ├── language
│   │   ├── embedding_weights.pkl
│   │   ├── idx2word.pkl
│   │   ├── vid2groundtruth.pkl
│   │   ├── vid2language.pkl
│   │   ├── word2idx.pkl
│   │   └── vid2fillmask_MSRVTT.pkl
│   ├── MSRVTT_splits
│   │   ├── MSRVTT_test_list.pkl
│   │   ├── MSRVTT_train_list.pkl 
│   │   └── MSRVTT_valid_list.pkl
│   └── visual
│       ├── MSRVTT_C3D_test.hdf5
│       ├── MSRVTT_C3D_train.hdf5
│       ├── MSRVTT_C3D_valid.hdf5
│       ├── MSRVTT_inceptionresnetv2_test.hdf5
│       ├── MSRVTT_inceptionresnetv2_train.hdf5
│       ├── MSRVTT_inceptionresnetv2_valid.hdf5
│       ├── MSRVTT_vg_objects_test.hdf5
│       ├── MSRVTT_vg_objects_train.hdf5
│       └── MSRVTT_vg_objects_valid.hdf5
└── MSVD
    ├── language
    │   ├── embedding_weights.pkl
    │   ├── idx2word.pkl
    │   ├── vid2groundtruth.pkl
    │   ├── vid2language.pkl
    │   ├── word2idx.pkl
    │   └── vid2fillmask_MSVD.pkl
    ├── MSVD_splits
    │   ├── MSVD_test_list.pkl
    │   ├── MSVD_train_list.pkl
    │   └── MSVD_valid_list.pkl
    └── visual
        ├── MSVD_C3D_test.hdf5
        ├── MSVD_C3D_train.hdf5
        ├── MSVD_C3D_valid.hdf5
        ├── MSVD_inceptionresnetv2_test.hdf5
        ├── MSVD_inceptionresnetv2_train.hdf5
        ├── MSVD_inceptionresnetv2_valid.hdf5
        ├── MSVD_vg_objects_test.hdf5
        ├── MSVD_vg_objects_train.hdf5
        └── MSVD_vg_objects_valid.hdf5


Training & Testing
Training: MSR-VTT
python -u main.py --dataset_name MSRVTT --entity_encoder_layer 2 --entity_decoder_layer 4 --max_objects 9 --backbone_2d_name inceptionresnetv2 --backbone_2d_dim 1536 --backbone_3d_name C3D --backbone_3d_dim 2048 --object_name vg_objects --object_dim 2048 --max_epochs 25 --save_checkpoints_every 500 --data_dir data --model_name HMN --learning_rate 7e-5 --lambda_entity 0.1 --lambda_predicate 6.8 --lambda_sentence 7.2 --lambda_soft 3.5 --save_checkpoints_path checkpoints/MSRVTT/HMN_MSRVTT_model.ckpt1

Training: MSVD
python -u main.py --dataset_name MSVD --entity_encoder_layer 2 --entity_decoder_layer 2 --max_objects 8 --backbone_2d_name inceptionresnetv2 --backbone_2d_dim 1536 --backbone_3d_name C3D --backbone_3d_dim 2048 --object_name vg_objects --object_dim 2048 --max_epochs 20 --save_checkpoints_every 500 --data_dir ./data --model_name HMN --learning_rate 9.5e-5 --lambda_entity 0.5 --lambda_predicate 0.35 --lambda_sentence 1.1 --lambda_soft 0.4 --save_checkpoints_path checkpoints/MSVD/HMN_MSVD_model.ckpt1

Testing MSR-VTT & MSVD
Comment out train_fn in main.py first.

model = train_fn(cfgs, cfgs.model_name, model, hungary_matcher, train_loader, valid_loader, device)
For MSR-VTT:

python -u main.py --dataset_name MSRVTT --entity_encoder_layer 2 --entity_decoder_layer 4 --max_objects 9 --backbone_2d_name inceptionresnetv2 --backbone_2d_dim 1536 --backbone_3d_name C3D --backbone_3d_dim 2048 --object_name vg_objects --object_dim 2048 --max_epochs 25 --save_checkpoints_every 500 --data_dir data --model_name HMN --learning_rate 7e-5 --lambda_entity 0.1 --lambda_predicate 6.8 --lambda_sentence 7.2 --lambda_soft 3.5 --save_checkpoints_path checkpoints/MSRVTT/HMN_MSRVTT_model.ckpt1

For MSVD:
python -u main.py --dataset_name MSVD --entity_encoder_layer 2 --entity_decoder_layer 2 --max_objects 8 --backbone_2d_name inceptionresnetv2 --backbone_2d_dim 1536 --backbone_3d_name C3D --backbone_3d_dim 2048 --object_name vg_objects --object_dim 2048 --max_epochs 20 --save_checkpoints_every 500 --data_dir ./data --model_name HMN --learning_rate 9.5e-5 --lambda_entity 0.5 --lambda_predicate 0.35 --lambda_sentence 1.1 --lambda_soft 0.4 --save_checkpoints_path checkpoints/MSVD/HMN_MSVD_model.ckpt1
