import torch
import torchvision.datasets as Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from generator import Generator, generator_loss, LATENT_SIZE
from discriminator import Discriminator, discriminator_loss

torch.manual_seed(2026)

device ='cuda'

auto = True
load = True

def main():

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize([100, 100]),
        v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    data = Dataset.CelebA(root='./data/', split='train', download=False, transform=transform)

    train_set = DataLoader(dataset=data, batch_size=16, shuffle=True, num_workers=10, pin_memory=True)

    print("Defining classes...")

    generator = Generator().to(device)
    print(next(generator.parameters()).device)
    discriminator = Discriminator().to(device)

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.00005)
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.00005)

    checkpoint = 39

    if load == True:
        generator.load_state_dict(torch.load(f"./versions/generator{checkpoint}.pt"))
        discriminator.load_state_dict(torch.load(f"./versions/discriminator{checkpoint}.pt"))

    def training_epoch(dataloader):
        generator.train()
        discriminator.train()
        sum_gloss=0.0
        sum_dloss=0.0
        sum_pred_true=0.0
        sum_pred_synth=0.0
        batches=0
        for x_true, attr in dataloader:
            attr = attr.to(device)
            pred_true=discriminator(x_true.to(device), attr)
            z=torch.randn(x_true.shape[0], LATENT_SIZE).to(device)
            x_synth=generator(z, attr)
            pred_synth=discriminator(x_synth, attr)

            optimizer_discriminator.zero_grad()
            dloss=discriminator_loss(pred_synth, pred_true)
            dloss.backward(retain_graph=True)
            optimizer_discriminator.step()

            pred_synth=discriminator(x_synth, attr)
            optimizer_generator.zero_grad()
            gloss=generator_loss(pred_synth)
            gloss.backward()
            optimizer_generator.step()

            sum_gloss += gloss.item()
            sum_dloss += dloss.item()
            sum_pred_true += pred_true.mean().item()
            sum_pred_synth += pred_synth.mean().item()
            batches += 1
            if batches%1000 == 0:
                print(f"{batches}/{len(dataloader)}")
        print('GLoss:', sum_gloss/batches,
            'DLoss:', sum_dloss/batches,
            'Pred_true:', sum_pred_true/batches,
            'Pred_synth:', sum_pred_synth/batches
            )
        
    print("Starting training phase...")
    for i in range(50):
        print("Epoch:", i)
        training_epoch(train_set)
        checkpoint = checkpoint + 1
        torch.save(generator.state_dict(), f"./versions/generator{checkpoint}.pt")
        torch.save(discriminator.state_dict(), f"./versions/discriminator{checkpoint}.pt")
        if auto == False:

            print("Continue?")
            val = input()
            if val != 'y':
                exit(0)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()