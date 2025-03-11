#  Towards zero-shot human-object interaction detection via vision-language integration
## Installation
git clone https://github.com/xwyscut/K2HOI.git
cd K2HOI

```
pip install -r requirements.txt
```

## Data preparation
This repo contains modified codes from HOICLIP, prepare the data according to [here] (https://github.com/Artanic30/HOICLIP.git)


## Pre-trained model

Download the pretrained model of DETR detector for [ResNet50](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth)
, and put it to the `params` directory.

```
python ./tools/convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2branch-hico.pth \
        --num_queries 64

python ./tools/convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2branch-vcoco.pth \
        --dataset vcoco \
        --num_queries 64
```

## Training

After the preparation, you can start training with the following commands.

### HICO-DET

```
# default setting
sh ./scripts/train_hico.sh
```

### V-COCO

```
sh ./scripts/train_vcoco.sh
```

### Zero-shot

```
# rare first unseen combination setting
sh ./scripts/train_hico_rf_uc.sh
# non rare first unseen combination setting
sh ./scripts/train_hico_nrf_uc.sh
# unseen object setting
sh ./scripts/train_hico_uo.sh
# unseen verb setting
sh ./scripts/train_hico_uv.sh
```

### Fractional data

```
# 50% fractional data
sh ./scripts/train_hico_frac.sh
```

### Generate verb representation for Visual Semantic Arithmetic

```
sh ./scripts/generate_verb.sh
```

We provide the generated verb representation in `./tmp/verb.pth` for hico and `./tmp/vcoco_verb.pth` for vcoco.

## Evaluation

### HICO-DET

You can conduct the evaluation with trained parameters for HICO-DET as follows.

```
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --use_env \
        main.py \
        --pretrained [path to your checkpoint] \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --eval \
        --zero_shot_type default \
        --with_clip_label \
        --with_obj_clip_label \
        --use_nms_filter
```

For the official evaluation (reported in paper), you need to covert the prediction file to a official prediction format
following [this file](./tools/covert_annot_for_official_eval.py), and then
follow [PPDM](https://github.com/YueLiao/PPDM) evaluation steps.

### V-COCO

Firstly, you need the add the following main function to the vsrl_eval.py in data/v-coco.

```
if __name__ == '__main__':
  import sys

  vsrl_annot_file = 'data/vcoco/vcoco_test.json'
  coco_file = 'data/instances_vcoco_all_2014.json'
  split_file = 'data/splits/vcoco_test.ids'

  vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)

  det_file = sys.argv[1]
  vcocoeval._do_eval(det_file, ovr_thresh=0.5)
```

Next, for the official evaluation of V-COCO, a pickle file of detection results have to be generated. You can generate
the file with the following command. and then evaluate it as follows.

```
python generate_vcoco_official.py \
        --param_path pretrained/VCOCO_GEN_VLKT_S.pth \
        --save_path vcoco.pickle \
        --hoi_path data/v-coco \
        --num_queries 64 \
        --dec_layers 3 \
        --use_nms_filter \
        --with_clip_label \
        --with_obj_clip_label

cd data/v-coco
python vsrl_eval.py vcoco.pickle

```

### Zero-shot

```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained [path to your checkpoint] \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers 3 \
        --eval \
        --with_clip_label \
        --with_obj_clip_label \
        --use_nms_filter \
        --zero_shot_type rare_first \
        --del_unseen
```