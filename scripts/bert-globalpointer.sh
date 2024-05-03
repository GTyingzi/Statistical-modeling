model_class='bert-globalpointer'
markup='span'

data_path='datasets/span_data/'
output_path='output/'

weight_decay=0.01
eps=1e-8
epochs=5
max_seq_len=512
seed=42
lr=1e-5
other_lr=2e-5

batch_size_train=4
batch_size_eval=256
eval_step=50

grad_acc_step=1
max_grad_norm=1
warmup_proportion=0.1

loss_type='mcc'
inner_dim=64

python run.py \
  --model_class ${model_class} \
  --markup ${markup} \
  --data_path ${data_path} \
  --output_path ${output_path} \
  --weight_decay ${weight_decay} \
  --eps ${eps} \
  --epochs ${epochs} \
  --max_seq_len ${max_seq_len} \
  --seed ${seed} \
  --lr ${lr} \
  --other_lr ${other_lr} \
  --batch_size_train ${batch_size_train} \
  --batch_size_eval ${batch_size_eval} \
  --eval_step ${eval_step} \
  --grad_acc_step ${grad_acc_step} \
  --max_grad_norm ${max_grad_norm} \
  --warmup_proportion ${warmup_proportion} \
  --loss_type ${loss_type} \
  --inner_dim ${inner_dim} \
  --do_train \
  --do_eval