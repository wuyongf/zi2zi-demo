## How to Use

```
git clone https://github.com/wuyongf/zi2zi-demo
cd zi2zi-demo/
mkdir fonts sample_dir binary_save_dir experiment_dir/data 
```

### Prerequisite - Prepare Your Own Source Font and Target Font
1. Find your favorite Chinese font at [foundertype](https://www.foundertype.com/index.php/FindFont/index)
2. Place the source fonts in the `./fonts/source/` directory (at least 1 font).
3. Place the target fonts in the `./fonts/target/` directory (at least 1 font).

###  Configure the Development Environment
```
cd zi2zi-demo/
pip install -r requirements.txt
```

### Create Samples (Paired Data)
```
cd zi2zi-demo/
python ttf2sample_imgs.py --src_font 仓耳今楷03-W03.ttf --sample_count 200
```

### Split Data into Training Set and Testing Set

```
python package.py --dir=sample_dir --save_dir=binary_save_dir --split_ratio=0.2
```

### Training

```
mv ./binary_save_dir/train.obj ./binary_save_dir/val.obj ./experiment_dir/data/
python train.py --experiment_dir=experiment_dir --gpu_ids=cuda:0 --input_nc=1 --batch_size=128 --epoch=3300 --sample_steps=200 --checkpoint_steps=2000
```
`description: `
During the training, you can see the inference samples under `./experiment_dir/sample/` to observe how the model gradually learns the Chinese font.

### Infer
```
python infer_interpolate_yf.py --experiment_dir experiment_dir --gpu_ids cuda:0 --batch_size 128 --resume 8000 --from_txt --src_font ./fonts/target/柳公权柳体.ttf --src_txt 深 --label 17 --result_folder 20230514 --gen_no 0 --infer_name 0
```

`description: `
You can check the inference result under `./experiment_dir/infer/<result_folder>`

## Citation
This is the modified verison of [zi2zi-pytorch](https://github.com/EuphoriaYan/zi2zi-pytorch), which is used for AI4Future project at CUHK.
