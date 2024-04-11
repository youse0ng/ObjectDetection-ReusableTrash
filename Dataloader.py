import torch
from torch.utils.data import DataLoader
from Dataset import TrashDataset
from pathlib import Path
from torchvision.transforms import v2

train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.Resize((224,224)),
    v2.RandomVerticalFlip(0.5)
])

classes=['c_1','c_2_01','c_2_02','c_3',
         'c_4_01_02','c_4_02_01_02',
         'c_4_02_02_02','c_4_02_03_02','c_4_03','c_5_02',
         'c_6','c_7','c_1_01','c_2_02_01',
         'c_3_01','c_4_03_01','c_5_01_01',
         'c_5_02_01','c_6_01','c_7_01',
         'c_4_01_01','c_4_02_01_01',
         'c_4_02_02_01','c_4_02_03_01',
         'c_5_01','c_8_01','c_8_02',
         'c_8_01_01','c_9']

def od_collate_fn(batch):
    '''
    Dataset에서 꺼내는 어노테이션 데이터의 크기는 화상마다 다름.
    화상 내의 물체 수가 두개이면 (2,5) 사이즈 [xmin, ymin, xmax, ymax, label]이 2개, 
    물체 수가 3개 이면 (3,5) 사이즈 [Xmin, Xmax, ymin, ymax, label]가 3개
    변화에 대응하는 DataLoader를 만드는 collate_fn을 작성
    collate_fn은 파이토치 리스트로 mini_batch를 작성하는 함수이다.
    '''
    targets=[]
    images=[]
    # batch = transformed_image, bbox , label
    for transformed_image, bbox, label in batch:
        images.append(transformed_image/255)
        targets.append({'boxes':torch.tensor(bbox), 'labels':torch.tensor(label)})
    
    return images,targets


if __name__ == '__main__':
    train_anno_path = Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Training\02.라벨링데이터')
    train_img_dir = Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Training\01.원천데이터')

    test_img_dir= Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Validation\01.원천데이터')
    test_anno_path= Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Validation\02.라벨링데이터')

    train_dataset = TrashDataset(classes,train_anno_path,train_img_dir,train_transforms)
    train_dataloader = DataLoader(train_dataset,2,True,collate_fn=od_collate_fn)
    images, targets = next(iter(train_dataloader))
    print(f'images : {images}')
    print(f'targets: {targets}')