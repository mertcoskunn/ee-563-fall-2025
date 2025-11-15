from ModelUNET import BalloonDataset, BallonSegmentator_Unet


if __name__ == "__main__":
    train_dataset = BalloonDataset("balloon_1/train")
    val_dataset = BalloonDataset("balloon_1/val")
    
    model_unet = BallonSegmentator_Unet()
    model_unet.train(train_dataset, val_dataset, epochs=40, lr=1e-4)