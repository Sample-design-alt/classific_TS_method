from glob import glob

datasets = glob('/data/root/data/chenrj/dataset/Multivariate_ts/*')
import os

for path in datasets:
    dataset = path.split('/')[-1]
    print(
        'python -u ./deep\\ learning/Time-Series-Library-main/run.py --task_name classification  --is_training 1 --root_path '
        '/data/root/data/chenrj/dataset/Multivariate_ts/{0}/ --model_id {0} --model ETSformer --data UEA '
        '--e_layers 3 --batch_size 16 --d_model 64 --d_ff 64 --top_k 3 --des \'Exp\' --itr 1 --learning_rate 0.001 '
        '--train_epochs 10 --patience 10 '
        .format(dataset),
        file=open('./deep learning/Time-Series-Library-main/scripts/classification/ETSformer_30.sh', 'a+'))
