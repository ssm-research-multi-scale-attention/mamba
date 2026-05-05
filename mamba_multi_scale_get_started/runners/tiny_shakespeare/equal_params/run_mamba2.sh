ROOT=$(pwd)/../../..

echo "Running mamba2 with depth 6 and block size 1024"
python $ROOT/code/train_lm.py \
    --config /home/jovyan/shares/SR008.fs2/nvidenisov/other/repos/mamba/mamba_multi_scale_get_started/configs/TinyShakespeare/tiny_shakespeare_mamba2.yaml \
    model.layer_headdims=[32,32,32,32,32,32] \
    data.block_size=1024 \
    experiment.name=tiny_shakespeare_mamba2_depth_6_block_size_1024

echo "Running mamba2 with depth 7 and block size 1024"
python $ROOT/code/train_lm.py \
    --config /home/jovyan/shares/SR008.fs2/nvidenisov/other/repos/mamba/mamba_multi_scale_get_started/configs/TinyShakespeare/tiny_shakespeare_mamba2.yaml \
    model.layer_headdims=[32,32,32,32,32,32,32] \
    data.block_size=1024 \
    experiment.name=tiny_shakespeare_mamba2_depth_7_block_size_1024
