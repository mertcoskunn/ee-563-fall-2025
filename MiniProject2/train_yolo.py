from ModelYOLO import BalloonYOLOTrainer

if __name__ == "__main__":
    trainer = BalloonYOLOTrainer(dataset_dir='balloon')
    trainer.train(epochs=30)