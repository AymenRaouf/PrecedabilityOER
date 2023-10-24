from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, roc_curve, auc, f1_score, recall_score
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
import time

torch.manual_seed(0)

def id_mapper(df_col, name):
    unique_id = df_col.unique()
    return pd.DataFrame({
            name : unique_id,
            'mappedID' : pd.RangeIndex(len(unique_id))
        })

def edge_construction(df1, df2, col, how, right_on, left_on = None):
    if left_on == None:
        left_on = right_on
    links = pd.merge(df1, df2, right_on = right_on, left_on = left_on, how = how)
    links = torch.from_numpy(links[col].values)
    return links


def train(model, train_data, epochs, lr, verbose = False):
    total_loss = 0
    loss_values = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    duration = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_data)
        ground_truth = train_data["OER", "before_sr", "OER"].edge_label
        assert pred.shape == ground_truth.shape, f'ERROR : Shapes differ between prediction and ground truth ! ({pred.shape, ground_truth.shape})'
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        loss_values.append(loss.item())
        if verbose and epoch % 10 == 0:
            print(f"Epoch : {epoch:03d}, Loss : {total_loss : .4f}")
    duration = time.time() - duration
    
    return {
        'Loss_values' : loss_values,
        'Loss' : total_loss,
        'Duration' : duration,
        'Model' : model
    }

def predict(model, test_data):

    preds = model(test_data).detach().numpy()
    preds_labels = (preds > 0.5) * 1
    ground_truths = test_data["OER", "before_sr", "OER"].edge_label
    assert preds.shape == ground_truths.shape, f'ERROR : Shapes differ between prediction and ground truth ! ({preds.shape, ground_truths.shape})'
    auc_score = roc_auc_score(ground_truths, preds)
    precision = precision_score(ground_truths, preds_labels, zero_division = np.nan)
    accuracy = accuracy_score(ground_truths, preds_labels)
    f1 = f1_score(ground_truths, preds_labels, average='macro')
    recall = recall_score(ground_truths, preds_labels, average='macro')
    return {
        'AUC' : auc_score,
        'Precision' : precision,
        'Accuracy' : accuracy,
        'Recall' : recall,
        'F1' : f1
    }