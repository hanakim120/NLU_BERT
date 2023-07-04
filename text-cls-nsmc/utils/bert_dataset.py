import torch
from torch.utils.data import Dataset 

class TextClassificationDataset(Dataset):
    # Pytorch의 Dataset을 상속받아 사용, 3개의 함수를 override해서 사용
    def __init__(self, texts, labels,tokenizer):
       
        input_texts = texts # list of texts
        targets = labels # list of labels
        
        self.inputs = []; self.labels = []
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt') 
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))
    
    def __len__(self):
        # 데이터셋 크기를 리턴
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 인덱스로 데이터를 얻을 수 있게 해주는 함수
        # 데이터를 미니 배치로 로딩하는 작업에서 __getitem__ 메서드는 각각의 미니 배치에 해당하는 데이터를 반환
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0), 
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0) 
        } 
