CUDA_VISIBLE_DEVICES=0 python3 eval_decoding.py \
    --dataset_path ./data/ZuCo/ \
    --model_path ./models/bert-base-uncased \
    --llm_path ./models/llama-2-7b-hf \
    --checkpoint_path ./data/ZuCo/checkpoints/decoding/best/task1_task2_task3_finetune_LLMTranslator_skipstep1_b256_20_30_0.0005_0.0005_unique_sent_EEG.pt \
    --config_path ./data/ZuCo/config/decoding/task1_task2_task3_finetune_LLMTranslator_skipstep1_b256_20_30_0.0005_0.0005_unique_sent_EEG.json  \
    --test_input EEG \
    --train_input EEG \




