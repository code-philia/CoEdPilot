lang=typescript
lr=5e-5
batch_size=10
source_length=512
target_length=512
data_dir=dataset
output_dir=model/$lang
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin
epochs=10
pretrained_model=microsoft/codebert-base
beam_size=10

CUDA_VISIBLE_DEVICES=0,1 nohup python run.py --do_test --model_type roberta --model_name_or_path $pretrained_model \
 --test_filename $test_file --output_dir $output_dir --load_model_path $test_model \
 --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size \
 --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs &
