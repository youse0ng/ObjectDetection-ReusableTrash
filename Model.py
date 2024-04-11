from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from Dataset import TrashDataset
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.transforms import v2
from Dataloader import od_collate_fn
import torch 
from tqdm.auto import tqdm

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

train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.Resize((224,224)),
    v2.RandomVerticalFlip(0.5)
])

train_anno_path = Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Training\02.라벨링데이터')
train_img_dir = Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Training\01.원천데이터')

test_img_dir= Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Validation\01.원천데이터')
test_anno_path= Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Validation\02.라벨링데이터')

train_dataset = TrashDataset(classes,train_anno_path,train_img_dir,train_transforms)
train_dataloader = DataLoader(train_dataset,2,True,collate_fn=od_collate_fn)

# model fine Tu
model1 = fasterrcnn_resnet50_fpn(pretrained = True)
num_classes = len(classes)+1
in_feature = model1.roi_heads.box_predictor.cls_score.in_features
model1.roi_heads.box_predictor = FastRCNNPredictor(in_feature , num_classes)

optimizer=torch.optim.Adam(model1.parameters(),lr=0.001)
for epochs in tqdm(range(10)):
    epoch_loss = 0 
    for batch, target in train_dataloader:
        loss_dict = model1(batch,target)
        loss = sum(v for v in loss_dict.values())
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(epoch_loss)