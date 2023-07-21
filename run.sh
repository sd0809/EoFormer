for loss_fn in 'DiceCE'
do
for lr in 2e-4
do
for wd in 1e-5
do

python -m torch.distributed.run --nproc_per_node=2 main.py \
--distributed \
--data_path BraTS2020 \
--dataset brats \
--modality T1_Ax T1_E_Ax T2_Ax T2_Flair_Ax \
--model eoformer \
--drop_path_rate 0.1 \
--n_seg_classes 3 \
--crop_H 128 \
--crop_W 128 \
--crop_D 128 \
--sw_batch_size 4 \
--inf_overlap 0.5 \
--loss_function $loss_fn \
--optim 'adamw' \
--lr_scheduler 'warmup_cosine' \
--learning_rate $lr \
--weight_decay $wd \
--num_workers 8 \
--batch_size 2 \
--n_epochs 300 \
--warmup_epochs 50 \
--val_every 10 \
--save_folder "/output" \
--manual_seed 4294967295 \
--test_seed 10 \
--gpu_id 0 1 \
--start_epoch 1 \

done
done
done