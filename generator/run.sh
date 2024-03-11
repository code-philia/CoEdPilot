lang=java
lr=5e-5
batch_size=20
source_length=512
target_length=128
data_dir=dataset
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/dev.jsonl
test_file=$data_dir/$lang/test.jsonl
epochs=4
pretrained_model=microsoft/codebert-base
beam_size=10

CUDA_VISIBLE_DEVICES=0,1 python run.py --do_train --do_eval --do_test --model_type roberta \
--model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --test_filename $test_file \
--output_dir $output_dir --max_source_length $source_length --max_target_length $target_length \
--beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size \
--learning_rate $lr --num_train_epochs $epochs
