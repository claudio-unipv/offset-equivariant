#!/usr/bin/env python3

import argparse
import torch
import tqdm.autonotebook as tqdm
import imgdata


def parse_args():
    parser = argparse.ArgumentParser("Train a model on CIFAR-10 images")
    a = parser.add_argument
    a("-b", "--batch-size", type=int, default=128, help="Images per minibatch")
    a("-r", "--data-dir", default="./data", help="Directory for the storage of images")
    a("-w", "--workers", type=int, default=2, help="Parallel workers")
    dev = ("cuda" if torch.cuda.is_available() else "cpu")
    a("-d", "--device", default=dev, help="Computing device")
    a("-l", "--learning-rate", type=float, default=0.1, help="Learning rate")
    a("--lr-schedule", type=int, nargs="+", default=[32000, 48000],
      help="Steps at which the learning rate is reduced")
    a("--lr-gamma", type=float, default=0.1, help="Reduction factor for the learning rate")
    a("--weight-decay", type=float, default=1e-4, help="Weight decay coefficient")
    a("--momentum", type=float, default=0.9, help="Momentum coefficient")
    a("-s", "--steps", type=int, default=64000, help="Training steps")
    a("-v", "--validate-every", type=int, default=1000, help="Validation frequency")
    a("-R", "--seed", type=int, default=25803, help="RNG seed")
    a("-S", "--save", help="Save model to file")
    a("-L", "--load", help="Load model from file")
    a("-D", "--depth", type=int, default=3, help="Model depth")
    a("-e", "--equivariant", action="store_true", help="Use the equivariant version of the model")
    a("--aug-saturation", type=float, default=0.0, help="Augmentation: saturation")
    a("--aug-hue", type=float, default=0.0, help="Augmentation: hue")
    a("--aug-contrast", type=float, default=0.0, help="Augmentation: contrast")
    a("--aug-brightness", type=float, default=0.0, help="Augmentation: brightness")
    return parser.parse_args()


def training(net, dataiter, optimizer, scheduler, steps, device):
    criterion = torch.nn.CrossEntropyLoss()
    n = 0
    tot_loss = 0.0
    tot_error = 0.0
    net.train()
    for images, labels in dataiter:
        if n == steps:
            break
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        tot_loss += loss.item()
        tot_error += (outputs.argmax(1) != labels).float().mean()
        n += 1
    return tot_loss / n, tot_error / n


def test_epoch(net, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    n = 0
    tot_loss = 0.0
    tot_error = 0.0
    net.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = net(images)
            loss = criterion(outputs, labels)
        tot_loss += loss.item()
        tot_error += (outputs.argmax(1) != labels).float().mean().item()
        n += 1
    return tot_loss / n, tot_error / n


def cycle(iterable):
    """Repeat the iterable."""
    while True:
        yield from iterable


def main():
    args = parse_args()
    for k, v in vars(args).items():
        print(f"{k:>16s}:  ", v)
    print()
    aug = {
        "aug_sat": args.aug_saturation,
        "aug_hue": args.aug_hue,
        "aug_bri": args.aug_brightness,
        "aug_con": args.aug_contrast
    }
    torch.manual_seed(args.seed)
    train_loader = imgdata.make_cifar_loader(True, args, 0, args.equivariant, **aug)
    test_loader = imgdata.make_cifar_loader(False, args, 0, args.equivariant)
    for ld, tag in ((train_loader, "Training set"), (test_loader, "Test set")):
        sz = "x".join(map(str, ld.dataset[0][0].size()))
        print(f"{tag}: {len(ld.dataset)} images {sz}")
    net = imgdata.make_model(args.depth, args.equivariant)
    net.to(args.device)
    ps = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {ps / 1000000:.2f}M")
    print()

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_schedule,
                                                     gamma=args.lr_gamma)
    if args.load:
        save_data = torch.load(args.load, map_location=args.device)
        net.load_state_dict(save_data["model_state_dict"])
        optimizer.load_state_dict(save_data["optimizer_state_dict"])
        scheduler.load_state_dict(save_data["scheduler_state_dict"])
        start_step = save_data["steps"]
    else:
        start_step = 0

    dataiter = cycle(train_loader)
    for step in tqdm.trange(start_step, args.steps, args.validate_every):
        msg0 = f"{step + args.validate_every:6d}"
        loss, error = training(net, dataiter, optimizer, scheduler, args.validate_every,
                               args.device)
        msg1 = f"  TRAIN  loss: {loss:.3f}  error: {error:.3f}"
        loss, error = test_epoch(net, test_loader, args.device)
        msg2 = f"  TEST  loss: {loss:.3f}  error: {error:.3f}"
        tqdm.tqdm.write(msg0 + msg1 + msg2)
        if args.save:
            save_data = {
                'steps': step + args.validate_every,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'error': error,
                'args': args
            }
            torch.save(save_data, args.save)


if __name__ == "__main__":
    main()
