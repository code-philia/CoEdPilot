lang=typescript #programming language
beam_size=10
batch_size=20
source_length=512
target_length=128
output_dir=model/$lang
data_dir=dataset
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

CUDA_VISIBLE_DEVICES=0,1 python run.py --do_test --model_type roberta \
--model_name_or_path microsoft/codebert-base --load_model_path $test_model \
--test_filename $test_file --output_dir $output_dir --max_source_length $source_length \
--max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size \
