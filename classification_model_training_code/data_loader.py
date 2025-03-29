import numpy as np
import pandas as pd
import os
import math
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize
import albumentations as album

def get_training_augmentation():
    train_transform = [
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        album.PadIfNeeded(min_height=256, min_width=256, always_apply=True),
    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)

class ArtClassDataloader:
    def __init__(
            self,
            images_dir,
            image_data,
            y_encoded,
            shape=(256, 256, 3),
            batch_size=16,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_data = image_data
        self.x = [os.path.join(images_dir, image_name) for image_name in image_data]
        self.y = y_encoded
        self.batch_size = batch_size
        self.shape = shape
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return int(math.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            batch_x = self.x[idx]
            batch_y = self.y[idx]
        else:
            low = int(idx * self.batch_size)
            high = int(min(low + self.batch_size, len(self.x)))
            batch_x = self.x[low:high]
            batch_y = self.y[low:high]
        
        x_return = []
        
        for file_name in batch_x:
            try:
                image = io.imread(file_name)
                if self.augmentation:
                    sample = self.augmentation(image=image)
                    image = sample['image']
                if self.preprocessing:
                    sample = self.preprocessing(image=image)
                    image = sample['image']
                x_return.append(image)
            except OSError:
                batch_y = np.delete(np.array(batch_y), batch_x.index(file_name), 0)

        return np.array([resize(x, self.shape) for x in x_return]), np.array(batch_y)

def prepare_labels(data_desrc_dir):
    labels_df = pd.read_csv(data_desrc_dir)
    labels_df = labels_df.dropna()
    labels_df['TUPLES'] = [(labels_df['TECHNIQUE'][i], labels_df['TYPE'][i], 
                          labels_df['SCHOOL'][i], labels_df['TIMELINE'][i]) 
                          for i in labels_df.index]
    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(labels_df['TUPLES'])
    return labels_df, y, binarizer

def create_dataloaders(labels_df, y, binarizer, art_dir, image_size, batch_size):
    X_train, X_test, y_train, y_test = train_test_split(
        labels_df, labels_df['TUPLES'],
        test_size=0.20,
        random_state=42
    )
    
    training_dataloader = ArtClassDataloader(
        images_dir=art_dir,
        image_data=X_train['FILE'].values,
        y_encoded=binarizer.transform(y_train),
        batch_size=batch_size,
        shape=image_size,
        augmentation=get_training_augmentation()
    )
    
    test_dataloader = ArtClassDataloader(
        images_dir=art_dir,
        image_data=X_test['FILE'].values,
        y_encoded=binarizer.transform(y_test),
        batch_size=batch_size,
        shape=image_size,
        augmentation=get_validation_augmentation()
    )
    
    return training_dataloader, test_dataloader