import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import tqdm
import matplotlib.pyplot as plt

from tod import make_dataset_with_index, SequentialWithArgs, TurnOverDropout

seed = 0
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

N_train = 3000
N_valid = 200


def calculate_influence_between_datasets(model, train_dataset, val_dataset):
    ### influence
    # out[i][j] is the influence on i-th validation instance from j-th training instance
    model.eval()
    influence_on_vals = []
    print("calculate influence on validation instance from each training instance")
    print("N_train =", len(train_dataset), "N_val =", len(val_dataset))
    for xs, ys in tqdm.tqdm(torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)):
        loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)
        influence_batches = []
        for _, _, train_indices in loader:
            with torch.no_grad():
                xs = xs.expand(train_indices.shape[0], xs.shape[1])
                preds_if_seen = model(xs, indices=train_indices, flips=False)
                loss_if_seen = nn.functional.cross_entropy(preds_if_seen, ys.expand(100), reduction="none")
                preds_if_unseen = model(xs, indices=train_indices, flips=True)
                loss_if_unseen = nn.functional.cross_entropy(preds_if_unseen, ys.expand(100), reduction="none")
                influence = loss_if_unseen - loss_if_seen
                # if high, the training instance is good
                # if low (negative), the training instance is bad
                influence_batches.append(influence)
        influence_on_vals.append(torch.cat(influence_batches, 0).detach())
    return torch.stack(influence_on_vals, 0).detach()

def prepare_dataset():
    # prepare dataset
    train_dataset = make_dataset_with_index(datasets.MNIST)(
        "/tmp/mnist", train=True,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x:x.reshape(-1)),
        ]),
        download=True)
    # subsampling for simple experiment
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [N_train, len(train_dataset)-N_train], generator=torch.Generator().manual_seed(41))

    val_dataset = datasets.MNIST(
        "/tmp/mnist", train=False,
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Lambda(lambda x:x.reshape(-1)),
        ]),
        download=True)
    # subsampling for simple experiment
    val_dataset, _ = torch.utils.data.random_split(
        val_dataset, [N_valid, len(val_dataset)-N_valid], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset

def prepare_dataset_noaug():
    # prepare dataset
    train_dataset = make_dataset_with_index(datasets.MNIST)(
        "/tmp/mnist", train=True,
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Lambda(lambda x:x.reshape(-1)),
        ]),
        download=True)
    # subsampling for simple experiment
    train_dataset, _ = torch.utils.data.random_split(
        train_dataset, [N_train, len(train_dataset)-N_train], generator=torch.Generator().manual_seed(41))
    return train_dataset

def prepare_model(use_turnover_dropout=True):
    # prepare model
    dim = 768
    model = SequentialWithArgs(
        nn.Linear(28*28, dim, bias=False),
        nn.ReLU(),
        TurnOverDropout(dim, p=0.5, seed=2000) if use_turnover_dropout else nn.Dropout(0.5),
        nn.Dropout(0.1),
        nn.Linear(dim, dim, bias=False),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(dim, dim, bias=False),
        nn.ReLU(),
        TurnOverDropout(dim, p=0.5, seed=3000) if use_turnover_dropout else nn.Dropout(0.5),
        nn.Dropout(0.1),
        nn.Linear(dim, 10, bias=False),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
    return model, optimizer

def evaluate(model, val_dataset):
    with torch.no_grad():
        model.eval()
        val_losses, val_accs, val_preds, val_GTs, val_imgs = [], [], [], [], []
        for batch in torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False):
            xs, ys = batch
            preds = model(xs)
            loss = nn.functional.cross_entropy(preds, ys)
            val_losses.append(loss.item())
            val_accs.append((torch.argmax(preds, -1) == ys).float().mean().item())
            val_preds.append(preds.cpu())
            val_GTs.append(ys.cpu())
            val_imgs.append(xs.cpu())
        print("valid acc={:.04f} loss={:.04f}".format(np.mean(val_accs), np.mean(val_losses)))
        model.train()
    return torch.cat(val_preds, dim=0), torch.cat(val_GTs, dim=0), torch.cat(val_imgs, dim=0).reshape(-1, 1, 28, 28)

def train(model, optimizer, train_dataset, val_dataset):
    ### training
    model.train()
    losses, accs = [], []
    for epoch in range(40):
        # training
        loader = tqdm.tqdm(torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True))
        for batch in loader:
            xs, ys, data_indices = batch  # [torch.Size([32, 784]), torch.Size([32]), torch.Size([32])]
            preds = model(xs, indices=data_indices)
            loss = nn.functional.cross_entropy(preds, ys, label_smoothing=0.01)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accs.append((torch.argmax(preds, -1) == ys).float().mean().item())
            loader.set_postfix(acc="{:.04f}".format(np.mean(accs[-20:])),
                               loss="{:.02f}".format(np.mean(losses[-20:])))
        print("train acc={:.04f} loss={:.04f}".format(np.mean(accs[-20:]), np.mean(losses[-20:])))
        evaluate(model, val_dataset)
    torch.save(model.state_dict(), "./mlp_chechpoint.pth")

if __name__ == "__main__":
    use_cache = True  # False if you retrain a model and recalculate influence

    # prepare model
    train_dataset, val_dataset = prepare_dataset()
    model, optimizer = prepare_model(use_turnover_dropout=True)
    if use_cache:
        model.load_state_dict(torch.load("./mlp_chechpoint.pth"))
    else:
        train(model, optimizer, train_dataset, val_dataset)
    model.eval()

    # evaluate and visualize
    val_preds, val_GTs, val_imgs = evaluate(model, val_dataset)
    vis_imgs = val_imgs + 0.3 * (val_preds.argmax(-1) != val_GTs)[:, None, None, None].float()  # vis wrong prediction
    tiles = torchvision.utils.make_grid(vis_imgs, nrow=10, padding=2, normalize=True)
    plt.imshow(tiles.permute(1, 2, 0))  # hwc
    plt.axis("off")
    plt.savefig("./predictions.png", dpi=200, bbox_inches='tight', pad_inches=0)

    # calculate influence validation-from-train
    if use_cache:
        influence_matrix = torch.tensor(np.load("influence_matrix.npy"))
    else:
        influence_matrix = calculate_influence_between_datasets(model, train_dataset, val_dataset)
        np.save("influence_matrix.npy", influence_matrix.cpu().numpy())
    print(influence_matrix.shape)  # (N_valid, N_train)

    # visualize bad/good training instances for each validation instance
    bad_vis_imgs, good_vis_imgs, GTs, preds, scores = [], [], [], [], []
    padding = torch.full((28, 4), 0.5)
    train_dataset = prepare_dataset_noaug()
    train_GTs = torch.tensor([d[1] for d in train_dataset])
    for i in range(len(val_dataset)):
        if val_preds[i].argmax() == val_GTs[i]:
            # show only results with wrong predictions
            continue
        preds.append(val_preds[i].argmax().item())
        GTs.append(val_GTs[i].item())

        score = ["{:.2f}".format(s) for s in torch.nn.functional.softmax(val_preds[i], dim=0).numpy().tolist()]
        scores.append(score)

        # show good training instances top3
        argsort_low_to_high = torch.argsort(influence_matrix[i])
        good_trains = [train_dataset[j][0] for j in argsort_low_to_high.numpy()[::-1][:3]]
        vis_imgs = [val_dataset[i][0]] + [padding] + good_trains
        vis_imgs = torch.cat([img.reshape(28, -1) for img in vis_imgs], dim=1)  # hw
        good_vis_imgs.append(vis_imgs)

        # show bad training instances with wrong label
        influence_matrix[i][train_GTs != val_preds[i].argmax()] += 10000.0  # ignore other labels
        argsort_low_to_high = torch.argsort(influence_matrix[i])
        bad_trains = [train_dataset[j][0] for j in argsort_low_to_high[:3]]
        vis_imgs = [val_dataset[i][0]] + [padding] + bad_trains
        vis_imgs = torch.cat([img.reshape(28, -1) for img in vis_imgs], dim=1)  # hw
        bad_vis_imgs.append(vis_imgs)

    bad_vis_imgs = torch.cat(bad_vis_imgs, dim=0)
    plt.imshow(bad_vis_imgs, cmap="gray")
    plt.axis("off")
    plt.savefig("./pred_and_bad_trains.png", dpi=200, bbox_inches='tight', pad_inches=0)

    good_vis_imgs = torch.cat(good_vis_imgs, dim=0)
    plt.imshow(good_vis_imgs, cmap="gray")
    plt.axis("off")
    plt.savefig("./pred_and_good_trains.png", dpi=200, bbox_inches='tight', pad_inches=0)

    open("gt_vs_pred.txt", "w").write("\n".join(f"{g}\t{p}\t{' '.join(s)}" for g, p, s in zip(GTs, preds, scores)))
