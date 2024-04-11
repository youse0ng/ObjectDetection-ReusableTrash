from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path
from torchvision.utils import draw_bounding_boxes
from json import load
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
import matplotlib.pyplot as plt

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

# TrashDataset에서 불러와야할 것들
'''전처리한 화상의 텐서 형식 데이터와 어노테이션 획득'''
'''전터리한 화상의 텐서 형식 데이터, 어노테이션, 화상의 높이, 폭 취득'''
train_anno_path = Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Training\02.라벨링데이터')
train_img_dir = Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Training\01.원천데이터')

test_img_dir= Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Validation\01.원천데이터')
test_anno_path= Path(r'C:\Users\hyssk\MyThesisProject\307.생활폐기물 데이터 활용ㆍ환류\01-1.정식개방데이터\Validation\02.라벨링데이터')

# Trash Dataset에서 뭘 뽑아내야할까
# Image의 [3 channels, width, height] 에 대한 정보, 해당 이미지의 Annotation 정보 (class 정보, BBOX)
class TrashDataset(Dataset):
    def __init__(self, classes, anno_path, img_dir, transform = None):
        self.anno_list = sorted(list(anno_path.glob('*.json'))) # annotation 파일들 모으고 sorted로 정렬
        self.img_list = sorted(list(img_dir.glob('*.jpg'))) # image 파일들을 모으고 sorted로 정렬
        self.transform= transform
        self.classes = classes

    def __getitem__(self, idx):
        # img 읽어오기
        img = read_image(str(self.img_list[idx]))

        # img와 맞는 annotation_path
        annotation_path = self.anno_list[idx]

        #  ret : [[xmin,ymin,xmax,ymax,label_ind],...]
        ret=[]
        with open(annotation_path, 'r',encoding='UTF8') as f:
            data = load(f)
            for object in data['objects']:
                bndbox=[]
                for coord in object["annotation"]['coord']: # [X,Y,Width,Height] -> [X,Y,Xmax,Ymax] 변환
                    if coord == 'x':
                        bndbox.append(int(object['annotation']['coord'][coord])) # / float(data['Info']['RESOLUTION'].split('/')[0]))
                    elif coord =='y':
                        bndbox.append(int(object['annotation']['coord'][coord])) # / float(data['Info']['RESOLUTION'].split('/')[1]))
                    elif coord == 'width':
                        bndbox.append(int(object['annotation']['coord']['x'] + object['annotation']['coord'][coord])) # / float(data['Info']['RESOLUTION'].split('/')[0]))
                    elif coord == 'height':
                        bndbox.append(int(object['annotation']['coord']['y'] + object['annotation']['coord'][coord])) # / float(data['Info']['RESOLUTION'].split('/')[1]))
                bndbox.append(classes.index(object['class_name']))
        # [3 channels, width, height], [[Xmin,Ymin,Xmax,Ymax,label_idx],....] // shape [N,5]
                width,height = data['Info']['RESOLUTION'].split('/')
                ret.append(bndbox)
        coord = [row[:-1] for row in ret]
        boxes = tv_tensors.BoundingBoxes(coord,format="XYXY",canvas_size=(int(height),int(width)))
        transformed_img, transformed_boxes = self.transform(torch.tensor(img),boxes)
        class_idx = [row[-1] for row in ret]
        return transformed_img, transformed_boxes, class_idx
    def __len__(self):
        return len(self.anno_list)
    

if __name__ == '__main__':
    train_transform = v2.Compose([
        v2.ToTensor(),
        v2.RandomRotation(90)
    ])

    train_dataset = TrashDataset(classes=classes,anno_path= train_anno_path,img_dir = train_img_dir,transform=train_transform)
    img,boxes,class_idx = train_dataset[0]
    drawn_box = draw_bounding_boxes(img, boxes, labels=[classes[i] for i in class_idx], colors=(255,0,0))

    plt.figure(figsize=(12,12))
    plt.imshow(drawn_box.permute(1,2,0))
    plt.axis(False)
    plt.show()
