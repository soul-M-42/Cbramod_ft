s_duration=10
# python preprocessing/preprocessing_faced_dynamic.py --s_duration $s_duration

# python preprocessing/preprocessing_faced.py --s_duration $s_duration

# python finetune_main.py --downstream_dataset FACED --datasets_dir ./data/datasets/BigDownstream/Faced/processed --num_of_classes 9 --s_duration $s_duration

python finetune_main.py --downstream_dataset FACED --datasets_dir ./data/datasets/BigDownstream/Faced/processed_dynamic --num_of_classes 9 --num_of_regression 6 --s_duration $s_duration