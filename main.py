if __name__ == "__main__":
    from dataset import basic_transform
    from trainer import Trainer
    from util import *
    import torch.optim as optim
    import torch.nn as nn
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    import wandb
    wandb.login()

    root = '/root/default/dgm-data/celeb/'
    model_path = '/root/default/dgm-yumin/ckpt/vqvae_1.ckpt'
    batch_size = 256
    resolution_size = 128
    image_size = (resolution_size, resolution_size)
    channel_size = 3
    learning_rate = 1e-2
    device = 'cuda'
    epochs = 30
    loss_fn = nn.MSELoss()

    config = {
        'Data': 'CelebA',
        'Batch size': batch_size,
        'Learning rate': learning_rate,
        'Epochs': epochs
    }

    wandb.init(
        project="DGM-PyTorch-Study",
        name="VQ-VAE-{}-{}".format(config['Data'], "yumin_1"),
        config=config,
    )
    
    train_data = datasets.CelebA(root, split='train', transform=basic_transform(image_size), download=True)
    val_data = datasets.CelebA(root, split='test', transform=basic_transform(image_size), download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)

    from VQVAE import VQVAE

    model = VQVAE(in_channels = channel_size, embedding_dim = 32, 
                 num_embeddings = 512, commitment_cost = 0.5)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader, device=device, model_path=model_path)
    trainer.fit(epochs)
    
    # model.load_state_dict(torch.load(model_path))
    # model.eval()

    x = next(iter(val_loader))
    
    x = x[0].to(device)
    sample_imgs, _ = model(x)
    save_images(sample_imgs.cpu(), 'Reconstructured CelebA Images')
        
    model.eval()
    sample_imgs = model.sample(64, device=device)
    save_images(sample_imgs.cpu(), 'Generated CelebA Images')
    print("Training completed.")