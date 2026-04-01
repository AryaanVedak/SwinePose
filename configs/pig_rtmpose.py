# ============================================================
# RTMPose-m: Single stage, 100 epochs
# Train: train_internal.json (2634 frames, 68 videos)
# Val:   val_internal.json   (465 frames, checkpoint selection)
# Test:  val_videosplit.json (679 frames, DO NOT USE DURING TRAINING)
# ============================================================

_base_ = ['_base_/default_runtime.py']

data_root = '/path/to/SwinePose'  # UPDATE: set to your SwinePose dataset root

dataset_info = dict(
    dataset_name='pig_pose',
    paper_info=dict(),
    keypoint_info={
        0:  dict(name='nose',               id=0,  color=[255,0,0],   type='upper', swap=''),
        1:  dict(name='front',              id=1,  color=[255,85,0],  type='upper', swap=''),
        2:  dict(name='right_carpal', id=2,  color=[255,170,0], type='upper', swap='left_carpal'),
        3:  dict(name='right_elbow',  id=3,  color=[255,255,0], type='upper', swap='left_elbow'),
        4:  dict(name='right_tarsal',       id=4,  color=[170,255,0], type='lower', swap='left_tarsal'),
        5:  dict(name='right_stifle',       id=5,  color=[85,255,0],  type='lower', swap='left_stifle'),
        6:  dict(name='hip',                id=6,  color=[0,255,0],   type='lower', swap=''),
        7:  dict(name='front_right_toe', id=7,  color=[0,255,85],  type='lower', swap='front_left_toe'),
        8:  dict(name='rear_right_toe',  id=8,  color=[0,255,170], type='lower', swap='rear_left_toe'),
        9:  dict(name='left_carpal',  id=9,  color=[0,255,255], type='upper', swap='right_carpal'),
        10: dict(name='left_elbow',   id=10, color=[0,170,255], type='upper', swap='right_elbow'),
        11: dict(name='front_left_toe',  id=11, color=[0,85,255],  type='lower', swap='front_right_toe'),
        12: dict(name='left_tarsal',        id=12, color=[0,0,255],   type='lower', swap='right_tarsal'),
        13: dict(name='left_stifle',        id=13, color=[85,0,255],  type='lower', swap='right_stifle'),
        14: dict(name='rear_left_toe',   id=14, color=[170,0,255], type='lower', swap='rear_right_toe'),
    },
    skeleton_info={
        0:  dict(link=('nose',               'front'),              color=[255,0,0]),
        1:  dict(link=('front',              'hip'),                color=[255,85,0]),
        2:  dict(link=('front',              'right_elbow'),  color=[255,170,0]),
        3:  dict(link=('right_elbow',  'right_carpal'), color=[255,255,0]),
        4:  dict(link=('right_carpal', 'front_right_toe'), color=[170,255,0]),
        5:  dict(link=('front',              'left_elbow'),   color=[85,255,0]),
        6:  dict(link=('left_elbow',   'left_carpal'),  color=[0,255,0]),
        7:  dict(link=('left_carpal',  'front_left_toe'),  color=[0,255,85]),
        8:  dict(link=('hip',                'right_stifle'),       color=[0,255,170]),
        9:  dict(link=('right_stifle',       'right_tarsal'),       color=[0,255,255]),
        10: dict(link=('right_tarsal',       'rear_right_toe'),  color=[0,170,255]),
        11: dict(link=('hip',                'left_stifle'),        color=[0,85,255]),
        12: dict(link=('left_stifle',        'left_tarsal'),        color=[0,0,255]),
        13: dict(link=('left_tarsal',        'rear_left_toe'),   color=[85,0,255]),
    },
    joint_weights=[1.0] * 15,
    sigmas=[0.072] * 15
)

# ── Codec ─────────────────────────────────────────────────────────────────
codec = dict(
    type='SimCCLabel',
    input_size=(192, 256),
    sigma=(4.9, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# ── Model ─────────────────────────────────────────────────────────────────
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth')),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=15,
        input_size=codec['input_size'],
        in_featuremap_size=(6, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.0,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# ── Pipelines ─────────────────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform',
         shift_factor=0.0,
         scale_factor=[0.75, 1.25],
         rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]

# ── Dataloaders ───────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/train_internal.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
        metainfo=dataset_info))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/val_internal.json',
        bbox_file=None,
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        metainfo=dataset_info))

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/val_videosplit.json',
        bbox_file=None,
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        metainfo=dataset_info))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/val_internal.json',
    nms_mode='none')

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/val_videosplit.json',
    nms_mode='none',
    outfile_prefix=data_root + '/rtmpose_predictions_test')

# ── Single stage, 100 epochs ──────────────────────────────────────────────
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0))

param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=100, milestones=[60, 85], gamma=0.1, by_epoch=True)
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='coco/AP',
        rule='greater'))

randomness = dict(seed=42)
