from create_dataset import FoodyDataset
from models import SimpleCNN
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize,Compose
import torch.nn as nn
from torch.optim import SGD
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type=str, default="foody", help="Root of the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)
if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    transform = Compose([
        Resize((224, 224)),  
        ToTensor(),          
    ])

    train_dataset = FoodyDataset(args.root,train= True, transform= transform)
    train_dataloader = DataLoader(
    dataset= train_dataset,
    batch_size= args.batch_size,
    num_workers= 4,
    shuffle= True,
    drop_last= True
    )

    test_dataset = FoodyDataset(args.root, train= False, transform= transform)
    test_dataloader = DataLoader(
    dataset= test_dataset,
    batch_size= args.batch_size,
    num_workers= 4,
    shuffle= False,
    drop_last= False
    )
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    writer = SummaryWriter("tensorboard")
    model = SimpleCNN(numclass=15).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr = 0.001, momentum= 0.9)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_accuracy = 0
    num_iter = len(train_dataloader)

    patience = 5
    counter = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour= "green")
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images) 
            loss_value = criterion(outputs, labels)
            progress_bar.set_description("epoch {}/{}. Interation {}/{}, Loss {:.3f}".format(epoch+1, args.epochs, iter +1, num_iter, loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch*num_iter +iter)

            #backward
            optimizer.zero_grad()   
            loss_value.backward()
            optimizer.step()
        
        model.eval()
        all_prediction = []
        all_label = []
        for iter, (images, labels) in enumerate(test_dataloader):
            all_label.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)
                indices= torch.argmax(predictions.cpu(), dim = 1) 
                all_prediction.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_label]
        all_predictions = [prediction.item() for prediction in all_prediction]
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions),class_names= test_dataset.category, epoch=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {} accuracy {}".format(epoch +1, accuracy))
        writer.add_scalar("Val/Accuracy", accuracy, epoch)

        checkpoint = {
            "epoch" : epoch + 1,
            "model" : model.state_dict(),
            "best_accuracy": best_accuracy,
            "optimizer" : optimizer.state_dict(),
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_accuracy:
            checkpoint = {
            "epoch" : epoch + 1,
            "best_accuracy" : accuracy,
            "model" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
        }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_accuracy = accuracy 
            counter = 0 
        else:
            counter +=1
        # if counter >= patience:
        #     print("Early stopping triggered!")
        #     break   