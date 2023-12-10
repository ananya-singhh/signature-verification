import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def fit(train_loader, val_loader, model, criterion, optimizer, scheduler, n_epochs, cuda, log_interval, threshold, log_dir, f,
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    loss_writer = SummaryWriter('runs/' + log_dir + "_loss")
    f1_writer = SummaryWriter('runs/' + log_dir + "_f1")
    tfpn_writer = SummaryWriter('runs/' + log_dir + "_tfpn")
    acc_writer = SummaryWriter('runs/' + log_dir + "_acc")

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_preds, train_targets, train_loss, train_tp, train_tn, train_fp, train_fn, train_f1, train_acc = train_epoch(train_loader, model, criterion, optimizer, log_interval, threshold, loss_writer, f1_writer, tfpn_writer, acc_writer, f)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        message += f'\tTP: {train_tp}, TN: {train_tn}, FP: {train_fp}, FN: {train_fn}, F1 Score: {train_f1}, Accuracy: {train_acc}'

        val_preds, val_targets, val_loss, val_tp, val_tn, val_fp, val_fn, val_f1, val_acc = test_epoch(val_loader, model, criterion, threshold, loss_writer, f1_writer, tfpn_writer, acc_writer)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        message += f'\tTP: {val_tp}, TN: {val_tn}, FP: {val_fp}, FN: {val_fn}, F1 Score: {val_f1}, Accuracy: {val_acc}'

        # loss_writer.add_scalar('Loss/Train', train_loss, epoch)
        # loss_writer.add_scalar('Loss/Test', val_loss, epoch)

        # f1_writer.add_scalar('F1Score/Train', train_f1, epoch)
        # f1_writer.add_scalar('F1Score/Test', val_f1, epoch)

        # acc_writer.add_scalar('Accuracy/Train', train_acc, epoch)
        # acc_writer.add_scalar('Accuracy/Test', val_acc, epoch)

        # tfpn_writer.add_scalar('Train/TP', train_tp, epoch)
        # tfpn_writer.add_scalar('Train/TN', train_tn, epoch)
        # tfpn_writer.add_scalar('Train/FP', train_fp, epoch)
        # tfpn_writer.add_scalar('Train/FN', train_fn, epoch)
        # tfpn_writer.add_scalar('Test/TP', val_tp, epoch)
        # tfpn_writer.add_scalar('Test/TN', val_tn, epoch)
        # tfpn_writer.add_scalar('Test/FP', val_fp, epoch)
        # tfpn_writer.add_scalar('Test/FN', val_fn, epoch)

        np.save(log_dir + '_predictions.npy', val_preds)
        np.save(log_dir + '_targets.npy', val_targets)

        print(message, file=f)

        scheduler.step()
    
    loss_writer.close()
    f1_writer.close()
    tfpn_writer.close()
    acc_writer.close()

def produce_label(output1, output2, threshold):
    # Calculate a similarity metric, such as cosine similarity
    similarity = nn.CosineSimilarity(dim=1, eps=1e-6)(output1, output2)

    # print(similarity)
    
    # Assign a label based on the similarity and a threshold
    label = torch.where(similarity > threshold, 1, 0)

    #label = 1 if similarity > threshold else 0
    
    return label.cpu()

def calculate_metrics(predictions, targets):
    tp = np.sum(np.logical_and(predictions == 1, targets == 1))
    tn = np.sum(np.logical_and(predictions == 0, targets == 0))
    fp = np.sum(np.logical_and(predictions == 1, targets == 0))
    fn = np.sum(np.logical_and(predictions == 0, targets == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    acc = (tp + tn) / (tp + fp + tn + fn)

    return tp, tn, fp, fn, f1, acc

def train_epoch(train_loader, model, criterion, optimizer, log_interval, threshold, loss_writer, f1_writer, tfpn_writer, acc_writer, f):
    model.train()
    losses = []
    total_loss = 0
    all_predictions = []
    all_targets = []

    for batch_idx, sample in enumerate(train_loader):
        input1 = sample['image0'].to('cuda')
        input2 = sample['image1'].to('cuda')
        label = sample['label'].to('cuda')

        optimizer.zero_grad()
        output1, output2 = model(input1, input2)

        predictions = produce_label(output1, output2, threshold)
        all_predictions = np.append(all_predictions, predictions)
        all_targets = np.append(all_targets, label.cpu())

        loss = criterion(output1, output2, label)
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        tp, tn, fp, fn, f1, acc = calculate_metrics(np.array(predictions), np.array(label.cpu()))

        loss_writer.add_scalar('Loss/Train', loss.item(), batch_idx)

        f1_writer.add_scalar('F1Score/Train', f1, batch_idx)

        acc_writer.add_scalar('Accuracy/Train', acc, batch_idx)

        tfpn_writer.add_scalar('Train/TP', tp, batch_idx)
        tfpn_writer.add_scalar('Train/TN', tn, batch_idx)
        tfpn_writer.add_scalar('Train/FP', fp, batch_idx)
        tfpn_writer.add_scalar('Train/FN', fn, batch_idx)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(input1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))

            print(message, file=f)
            losses = []
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    tp, tn, fp, fn, f1, acc = calculate_metrics(all_predictions, all_targets)

    total_loss /= (batch_idx + 1)
    return all_predictions, all_targets, total_loss, tp, tn, fp, fn, f1, acc

def test_epoch(val_loader, model, criterion, threshold, loss_writer, f1_writer, tfpn_writer, acc_writer):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        for batch_idx, sample in enumerate(val_loader):
            input1 = sample['image0'].to('cuda')
            input2 = sample['image1'].to('cuda')
            label = sample['label'].to('cuda')

            output1, output2 = model(input1, input2)
            
            predictions = produce_label(output1, output2, threshold)
            all_predictions = np.append(all_predictions, predictions)
            all_targets = np.append(all_targets, label.cpu())

            loss = criterion(output1, output2, label)
            val_loss += loss.item()

            tp, tn, fp, fn, f1, acc = calculate_metrics(np.array(predictions), np.array(label.cpu()))

            loss_writer.add_scalar('Loss/Test', loss.item(), batch_idx)

            f1_writer.add_scalar('F1Score/Test', f1, batch_idx)

            acc_writer.add_scalar('Accuracy/Test', acc, batch_idx)

            tfpn_writer.add_scalar('Test/TP', tp, batch_idx)
            tfpn_writer.add_scalar('Test/TN', tn, batch_idx)
            tfpn_writer.add_scalar('Test/FP', fp, batch_idx)
            tfpn_writer.add_scalar('Test/FN', fn, batch_idx)

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    tp, tn, fp, fn, f1, acc = calculate_metrics(all_predictions, all_targets)

    return all_predictions, all_targets, val_loss, tp, tn, fp, fn, f1, acc