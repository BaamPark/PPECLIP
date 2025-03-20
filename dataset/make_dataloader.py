from dataset.PPEDataset import PPEmultilabelDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import importlib
from utils.config import get_config

global_cfg = get_config()

def make_dataloader():
    dataset_cfg = global_cfg["DATASET"]
    hparam_cfg = global_cfg["HYPERPARAM"]
    
    train_transform = T.Compose([
            T.Resize(dataset_cfg["INPUT_IMG_SIZE"]),
            T.RandomHorizontalFlip(p=dataset_cfg["PROB"]),
            T.RandomCrop(dataset_cfg["INPUT_IMG_SIZE"]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    

    val_transform = T.Compose([T.Resize(dataset_cfg["INPUT_IMG_SIZE"]), 
                        T.ToTensor(), 
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

    module = importlib.import_module('dataset.PPEDataset')
    dataset_cls = getattr(module, dataset_cfg["NAMES"])
    train_data = dataset_cls(dataset_cfg["TRAIN_LABEL_PATH"],train_transform)
    val_data = dataset_cls(dataset_cfg["VAL_LABEL_PATH"],val_transform)
    test_data = dataset_cls(dataset_cfg["TEST_LABEL_PATH"],val_transform)

    # if dataset_cfg["NAMES"] == "PPEmultilabelDataset":
    #     train_data = PPEmultilabelDataset(dataset_cfg["TRAIN_LABEL_PATH"],train_transform)
    #     val_data = PPEmultilabelDataset(dataset_cfg["VAL_LABEL_PATH"],val_transform)
    #     test_data = PPEmultilabelDataset(dataset_cfg["TEST_LABEL_PATH"],val_transform)

    train_loader = DataLoader(
            train_data,
            batch_size=hparam_cfg["BATCH_SIZE"],
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
    
    val_loader = DataLoader(
            val_data,
            batch_size=hparam_cfg["BATCH_SIZE"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    
    test_loader = DataLoader(
            test_data,
            batch_size=hparam_cfg["BATCH_SIZE"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
    
    return train_loader, val_loader, test_loader