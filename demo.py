from utils.config import Config, get_config
Config.set_config_file("config/12cls_nosplit.yaml")

from dataset.make_dataloader import make_dataloader

def main():
    cfg = get_config()

    train_loader, val_loader, test_loader = make_dataloader()
    for img, lbl in train_loader:
        print('heelo')

if __name__ == "__main__":
    main()