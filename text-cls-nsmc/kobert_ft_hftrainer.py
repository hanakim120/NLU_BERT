# 패키지 설치
import argparse
import random
from sklearn.metrics import accuracy_score
import torch
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments

from utils.bert_dataset import TextClassificationDataset
from transformers import DataCollatorWithPadding
from utils.utils import read_text


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True) # 저장될 모델 파일 이름
    p.add_argument('--train_fn', required=True) # 학습에 사용될 파일 이름
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base') # 사전 학습된 모델 이름
    #  pretrained_model_name 예시:
    # - beomi/kcbert-base
    # - beomi/kcbert-large
    p.add_argument('--valid_ratio', type=float, default=.2) # valid set 비율
    p.add_argument('--batch_size_per_device', type=int, default=32) # device 당 batch size
    p.add_argument('--n_epochs', type=int, default=5) # epoch 수
    p.add_argument('--warmup_ratio', type=float, default=.2) # warmup 비율
    p.add_argument('--max_length', type=int, default=100) # 최대 길이

    config = p.parse_args()

    return config


def get_datasets(fn, tokenizer, valid_ratio=.2):
     # 데이터 파일을 읽어서 labels와 texts list 받기
    labels, texts = read_text(fn) 

    # label을 index로 바꾸기 위한 과정
    unique_labels = list(set(labels)) # 중복 제거
    label_to_index = {} # label을 index로 바꾸기 위한 dict
    index_to_label = {} # index를 label로 바꾸기 위한 dict

    for i, label in enumerate(unique_labels): 
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # texts와 labels를 zip으로 묶고, random.shuffle로 섞기
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    
    # 섞인 데이터를 다시 풀어서 texts와 labels로 나누기
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    
    # valid set의 비율에 따라서 index 설정
    idx = int(len(texts) * (1 - valid_ratio))
    
    # train set과 valid set으로 나누기

    train_dataset = TextClassificationDataset(texts[:idx], labels[:idx], tokenizer) # 앞에서부터 idx까지
    valid_dataset = TextClassificationDataset(texts[idx:], labels[idx:], tokenizer) # idx부터 끝까지
    
    
    return train_dataset, valid_dataset, index_to_label


def main(config):
    # pretrain된 tokenizer 불러오기
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
    
    # datasets과 index_to_label 받기
    train_dataset, valid_dataset, index_to_label = get_datasets(
        config.train_fn,
         tokenizer,
        valid_ratio=config.valid_ratio,    
    )
    
    print(
        '## train ## =', len(train_dataset),
        '## valid ## =', len(valid_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count() # 전체 batch size
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs) # 전체 iteration 수
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio) # warmup step 수
    
    print(
        '## total_iters ## =', n_total_iterations,
        '## warmup_iters ## =', n_warmup_steps,
    )

    # pretrained_model_name에 따라서 model loader를 다르게 설정
    model_loader = BertForSequenceClassification
    # pretrained model 불러오기
    model = model_loader.from_pretrained( 
        config.pretrained_model_name, # pretrained model 이름
        num_labels=len(index_to_label) # output label 개수
    )

    training_args = TrainingArguments(
        output_dir='./.checkpoints', # checkpoint 저장 경로
        num_train_epochs=config.n_epochs, # epoch 수
        per_device_train_batch_size=config.batch_size_per_device, # device 당 train batch size
        per_device_eval_batch_size=config.batch_size_per_device, # device 당 eval batch size
        warmup_steps=n_warmup_steps, # warmup step 수
        weight_decay=0.01, # weight decay
        fp16=True, # AMP 사용 여부
        evaluation_strategy='epoch', # epoch 단위로 eval
        save_strategy = 'epoch',
        logging_steps=n_total_iterations // 100, # logging step 수
        #save_steps=n_total_iterations // config.n_epochs,# save step 수
        load_best_model_at_end=True, # best model 불러오기 여부
    )

    def compute_metrics(pred):
        # prediction과 label을 받아서 accuracy 계산
        labels = pred.label_ids # np.ndarray
        # pred.predictions : label 예측값 (np.ndarray)
        preds = pred.predictions.argmax(-1) # 가장 높은 확률을 가진 label 예측값의 index

        return {
            'accuracy': accuracy_score(labels, preds)
        }
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # trainer 설정
    trainer = Trainer(
        model=model, 
        args=training_args,
        # # data_collator: batch를 만들 때 어떻게 할지 설정
        # data_collator=TextClassificationCollator(tokenizer, 
        #                                config.max_length, 
        #                                with_text=True), # text는 필요 없으므로 False
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics, # metric 계산
    )
    # train 시작
    trainer.train()

    torch.save({
        'bert': trainer.model.state_dict(), # best model의 weight 저장
        'config': config,
        'vocab': None,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
