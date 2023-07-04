def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            
            if line.strip() != '': # line이 공백이 아닐 때
                # 파일은 tab으로 구분, 첫번째 column은 label, 두번째 column은 text
                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]

    return labels, texts
