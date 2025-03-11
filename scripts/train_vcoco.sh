EXP_DIR=exps/vcoco

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
            --nproc_per_node=4 \
            --master_port 29515 \
            --use_env \
            main.py \
            --output_dir ${EXP_DIR} \
            --dataset_file vcoco \
            --hoi_path data/v-coco \
            --num_obj_classes 81 \
            --num_verb_classes 29 \
            --backbone resnet50 \
            --num_queries 64 \
            --dec_layers 3 \
            --epochs 90 \
            --lr_drop 60 \
            --use_nms_filter \
            --fix_clip \
            --batch_size 8 \
            --pretrained params/detr-r50-pre-2branch-vcoco.pth \
            --with_clip_label \
            --with_obj_clip_label \
            --gradient_accumulation_steps 1 \
            --num_workers 8 \
            --opt_sched "multiStep" \
            --dataset_root GEN \
            --model_name SWIFTHOICLIP \
            --zero_shot_type default \
            --resume ${EXP_DIR}/checkpoint_last.pth \
            --verb_pth ./tmp/vcoco_verb.pth \
            --with_rec_loss \
            --verb_loss_type "focal" \
            --image_verb_loss_type "asl" \
            --verb_embed_norm \
            --cat_specific_fc \
            --i_hidden_dim 512 \
            --interaction_decoder \
            --image_verb_loss \
            --hoi_text \
            --x_improved \
            --verb_weight 0.1 \
            --training_free_enhancement_path \
            ./training_free_ehnahcement/
