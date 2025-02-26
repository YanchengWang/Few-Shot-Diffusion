# generate_graph_embbedding
## Usage
Train diffusion on cora embedding data with 64 dimensions, run this command to launch train.py with your_dir
```
python  train.py --batchsize 256 --modch 64 --moddir your_dir --samdir your_dir --epoch 3000 --interval 30 --intervalplot 1 --nettype unet_1d --inch 1 --outch 1 --inputsize 64 --clsnum 7 --datatype gclemb --datadir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_embs.npy --labeldir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_ps_labels.npy --genum 140 --genbatch 140  --norm 1 
```

Generate graph embedding data, run this command to launch sample.py with pretrained model. 
You can set your_dir as /data-drive/backup/changyu/expe/gge/unet_1d_core64_all_norm to use our pretrained model
```
python sample.py --genum 70 --genbatch 70 --modch 64 --moddir your_dir --samdir your_dir --epoch 1200 --ddim yes --num_steps 50 --nettype unet_1d --inch 1 --outch 1 --inputsize 64 --clsnum 7 --datadir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_embs.npy --labeldir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora_64d/all_ps_labels.npy --norm 1
```

## train vae
```
/home/local/ASUAD/changyu2/miniconda3/envs/ldm/bin/python /home/local/ASUAD/changyu2/generate_graph_embbedding/train_vae.py --moddir /data-drive/backup/changyu/expe/gge/vae_512_256_64_mse_ema_kl0 --samdir /data-drive/backup/changyu/expe/gge/vae_512_256_64_mse_ema_kl0 --epoch 3000 --datadir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_embs.npy --labeldir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_ps_labels.npy --norm 1 --hidden_sizes 512 256 64 --intervalplot 30  --interval 30 --coef_kl 0
```

## encode embedding to latents
```
/home/local/ASUAD/changyu2/miniconda3/envs/ldm/bin/python /home/local/ASUAD/changyu2/generate_graph_embbedding/train_vae.py --moddir /data-drive/backup/changyu/expe/gge/vae_512_256_64_mse_ema_kl0 --samdir /data-drive/backup/changyu/expe/gge/vae_512_256_64_mse_ema_kl0 --lastepo 3000 --datadir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_embs.npy --labeldir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_ps_labels.npy --norm 1 --hidden_sizes 512 256 64 --run_type encode 
```

## decode latents to embedding
```
/home/local/ASUAD/changyu2/miniconda3/envs/ldm/bin/python /home/local/ASUAD/changyu2/generate_graph_embbedding/train_vae.py --moddir /data-drive/backup/changyu/expe/gge/vae_512_256_64_mse_ema_kl0 --samdir /data-drive/backup/changyu/expe/gge/vae_512_256_64_mse_ema_kl0 --lastepo 3000 --datadir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_embs.npy --labeldir /home/local/ASUAD/changyu2/generate_graph_embbedding/gcl_embeddings/cora/all_data/all_ps_labels.npy --norm 1 --hidden_sizes 512 256 64 --run_type decode --datadecode_dir /data-drive/backup/changyu/expe/gge/unet_1d_core64_encode_all_norm_ema/samples_8190_diffusion_3000_1.8.npy
```

## generated data
We generate 8190 cora 64-d embedding data(with GCL 64 embedding directly)
```
/data-drive/backup/changyu/expe/gge/unet_1d_core64_all_norm_ema/labels_8190_diffusion_3000_1.8.npy
/data-drive/backup/changyu/expe/gge/unet_1d_core64_all_norm_ema/samples_8190_diffusion_3000_1.8.npy
```

We generate 8190 cora 512-d embedding data(with GCL 512 embedding directly)
```
/data-drive/backup/changyu/expe/gge/unet_1d_core512_all_norm_ema/labels_2730_diffusion_3000_1.8.npy
/data-drive/backup/changyu/expe/gge/unet_1d_core512_all_norm_ema/samples_2730_diffusion_3000_1.8.npy
```

We generate 8190 cora 512-d embedding data (with VAE encoding and decoding)
```
/data-drive/backup/changyu/expe/gge/vae_512_256_64_mse_ema_kl0/latents_8190_vae_3000_512_decode.npy          (synthetic embeddings)
/data-drive/backup/changyu/expe/gge/unet_1d_core64_encode_all_norm_ema/labels_8190_diffusion_3000_1.8.npy    (synthetic labels)
```


We generate 8190 cora 512-d embedding data (with graph VAE encoding and decoding)
```
/data-drive/backup/changyu/expe/gge/graphvae_freeze_enc_feat_map1_lr2.4/latents_8190_gvae_50000_64_decode_feat.npy          (synthetic embeddings)
/data-drive/backup/changyu/expe/gge/unet_1d_core64_gvae_encode_all_norm_ema/labels_8190_diffusion_3000_1.8.npy              (synthetic labels)
/data-drive/backup/changyu/expe/gge/graphvae_freeze_enc_feat_map1_lr2.4/latents_8190_gvae_50000_64_decode_map.npy           (synthetic map)
```

We generate 19110 cora 512-d embedding data (with graph VAE encoding and decoding)
```
/data-drive/backup/changyu/expe/gge/graphvae_freeze_enc_feat_map1_lr2.4/latents_19110_gvae_50000_64_decode_feat.npy         (synthetic embeddings)
/data-drive/backup/changyu/expe/gge/unet_1d_core64_gvae_encode_all_norm_ema/labels_19110_diffusion_3000_1.8.npy             (synthetic labels)
/data-drive/backup/changyu/expe/gge/graphvae_freeze_enc_feat_map1_lr2.4/latents_19110_gvae_50000_64_decode_map.npy          (synthetic map)
```

We generate 33000 citeseer 512-d embedding data (with graph VAE encoding and decoding)
```
/data-drive/backup/changyu/expe/gge/graphvae_citeseer_freeze_enc_feat_map1_lr2.4/citeseer_latents_33000_gvae_10000_64_decode_feat.npy         (synthetic embeddings)
/data-drive/backup/changyu/expe/gge/unet_1d_citeseer64_gvae_encode_all_norm_ema/labels_33000_diffusion_3000_1.8.npy                           (synthetic labels)
/data-drive/backup/changyu/expe/gge/graphvae_citeseer_freeze_enc_feat_map1_lr2.4/citeseer_latents_33000_gvae_10000_64_decode_map.npy          (synthetic map)
```

We generate 198000 pubmed 512-d embedding data (with graph VAE encoding and decoding)
```
/data-drive/backup/changyu/expe/gge/graphvae_pubmed_freeze_enc_feat_map1_lr2.4/pubmed_latents_198000_gvae_20000_64_decode_feat.npy        (synthetic embeddings)
/data-drive/backup/changyu/expe/gge/unet_1d_pubmed64_gvae_encode_all_norm_ema/labels_198000_diffusion_3000_1.8.npy                        (synthetic labels)
/data-drive/backup/changyu/expe/gge/graphvae_pubmed_freeze_enc_feat_map1_lr2.4/pubmed_latents_198000_gvae_20000_64_decode_map.npy         (synthetic map)
```

We generate 190000 coauthor 512-d embedding data (with graph VAE encoding and decoding)
```
/data-drive/backup/changyu/expe/gge/graphvae_coauthor_freeze_enc_feat_map1_lr2.4/cs_latents_180000_gvae_16000_64_decode_feat.npy          (synthetic embeddings)
/data-drive/backup/changyu/expe/gge/unet_1d_coauthor64_gvae_encode_all_norm_ema/labels_180000_diffusion_3000_1.8.npy                      (synthetic labels)
/data-drive/backup/changyu/expe/gge/graphvae_coauthor_freeze_enc_feat_map1_lr2.4/cs_latents_180000_gvae_16000_64_decode_map.npy           (synthetic map)
```

We generate 27300 cora 512-d embedding data (with large graph VAE(for large scale dataset) encoding and decoding )
```
/data-drive/backup/changyu/expe/gge/large_graphvae3_freeze_enc_feat_nmap_3layer__feat1_nmap1_cmap1_lr2.4/latents_27300_lgvae_5000_64_decode_feat.npy    (synthetic embeddings)
/data-drive/backup/changyu/expe/gge/unet_1d_core64_large_gvae_encode_all_norm_ema/labels_27300_diffusion_2950_1.8.npy                                   (synthetic labels)
/data-drive/backup/changyu/expe/gge/large_graphvae3_freeze_enc_feat_nmap_3layer__feat1_nmap1_cmap1_lr2.4/latents_27300_lgvae_5000_64_decode_map.npy     (synthetic map)
```

## Unconditional Diffusion Model
Unconditional diffusion with 64d feature: 

[./unconditional_diffusion/ddpm_1d_length_64_dim_16.ipynb](./unconditional_diffusion/ddpm_1d_length_64_dim_16.ipynb)

Unconditional diffusion with PCA-generated feature:

[./unconditional_diffusion/ddpm_1d_PCA_length_16_dim_8.ipynb](./unconditional_diffusion/ddpm_1d_PCA_length_16_dim_8.ipynb)

[./unconditional_diffusion/ddpm_1d_PCA_length_64_dim_8.ipynb](./unconditional_diffusion/ddpm_1d_PCA_length_64_dim_8.ipynb)

[./unconditional_diffusion/ddpm_1d_PCA_length_64_dim_16.ipynb](./unconditional_diffusion/ddpm_1d_PCA_length_64_dim_16.ipynb)

[./unconditional_diffusion/ddpm_1d_PCA_length_128_dim_16.ipynb](./unconditional_diffusion/ddpm_1d_PCA_length_128_dim_16.ipynb)

## To-dos
**02/28 To-dos:**

(1) ~~Normalize the training data (Verified: Normalization is necessary. Add the data normalization to our pipeline.~~ Reference: [./unconditional_diffusion/ddpm_1d_length_64_dim_16.ipynb](./unconditional_diffusion/ddpm_1d_length_64_dim_16.ipynb))

(2) ~~Plot t-SNE for a small number of generated embeddings after each epoch.~~ (Code for t-sne can be found in [./tsne.ipynb](./tsne.ipynb)) 

(3) ~~Use PCA on the training data (to 32 d)~~ (Verified: Smaller dimension is needed for Cora.)

(4) ~~Reduce the number of parameters in Unet (Reduce the number of layers, Reduce the number of channels.)~~ (Not necessary for now)

**02/29 To-dos:**

(1) ~~Add EMA to our training pipeline~~. (You can use ema from [ema-pytorch](https://github.com/lucidrains/ema-pytorch). [This](https://github.com/ChangyuLiu2022/generate_graph_embbedding/blob/147ba128cf7e74ee48ea38863e26f95f522bc512/unconditional_diffusion/denoising-diffusion-pytorch-main/denoising_diffusion_pytorch/denoising_diffusion_pytorch_1d.py#L779) is a usage reference.)

(2) ~~Train the classifier-free conditional diffusion model on embedding with smaller dimensions.~~ You can find 64d Cora embeddings in [./gcl_embeddings/cora_64d](./gcl_embeddings/cora_64d). (You can also features of other dimension by applying PCA on the 512d Cora embeddings.)

(3) Train GCL encoder that generates low-dimensional (i.e. 32, 64, 128) features on Cora. (Training for 64d is already finished. Training for other dimensions is in progress.)

(4) (Later) Implement the metrics (i.e., FID and Inception Score) with a node embedding model so that we can compare the performance of different versions of our diffusion models. It can provide guidance for tuning the model architecture and hyper-parameters.

## Node Classification Results

#### Linear Classifier

64 d GCL Feature on Cora

| Synthetic Data Size (x Training Data Size) |  GCL  |   0   |  0.1  | 0.25  |  0.5  |   1   |   2   |   3   |   5   |  10   |
| :----------------------------------------: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|                  Test ACC                  | 0.831 | 0.814 | 0.824 | 0.837 | 0.839 | 0.839 | 0.827 | 0.825 | 0.823 | 0.817 |




