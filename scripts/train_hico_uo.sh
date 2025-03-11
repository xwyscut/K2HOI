#!/usr/bin/env bash
EXP_DIR=exps/hico/uo_2

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
            --nproc_per_node=4 \
            --master_port 29515 \
            --use_env \
            main.py \
            --output_dir ${EXP_DIR} \
            --dataset_file hico \
            --hoi_path data/hico_20160224_det \
            --num_obj_classes 80 \
            --num_verb_classes 117 \
            --backbone resnet50 \
            --num_queries 64 \
            --dec_layers 3 \
            --epochs 90 \
            --lr_drop 60 \
            --use_nms_filter \
            --fix_clip \
            --batch_size 8 \
            --pretrained params/detr-r50-pre-2branch-hico.pth \
            --with_clip_label \
            --with_obj_clip_label \
            --gradient_accumulation_steps 1 \
            --num_workers 8 \
            --opt_sched "multiStep" \
            --dataset_root GEN \
            --model_name HOICLIP \
            --del_unseen \
            --zero_shot_type unseen_object \
            --resume ${EXP_DIR}/checkpoint_last.pth \
            --verb_pth ./tmp/verb.pth \
            --verb_loss_type "focal" \
            --image_verb_loss_type "asl" \
            --verb_embed_norm \
            --cat_specific_fc \
            --i_hidden_dim 512 \
            --interaction_decoder \
            --image_verb_loss \
            --hoi_text \
            --x_improved \
            --with_rec_loss \
            --training_free_enhancement_path \
            ./training_free_ehnahcement/ 