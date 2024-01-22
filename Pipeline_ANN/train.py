import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional
import numpy as np
import pandas as pd
from nn.model import ANN
from util.utils import CustomDataset
from tqdm.auto import tqdm
import argparse
from eval.validation import *
from util.early_stop import EarlyStopper
from metric.rmsle import RMSLELoss, RMSELoss
from datasets.dataset import get_X, get_y
from metric.graph import get_graph
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import argparse

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def train(
  model:nn.Module,
  criterion:callable,
  optimizer:torch.optim.Optimizer,
  data_loader:DataLoader,
  device:str
) -> float:
  '''train one epoch
  
  Args:
      model: model
      criterion: loss
      optimizer: optimizer
      data_loader: data loader
      device: device
  '''
  model.train()
  total_loss = 0.
  for X, y in data_loader:
    X, y = X.to(device), y.to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * len(y)
  return total_loss/len(data_loader.dataset)

def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str,
  metric:Optional[torchmetrics.metric.Metric]=None,
) -> float:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
  '''
  model.eval()
  total_loss,correct = 0.,0.
  with torch.inference_mode():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      output = model(X)
      total_loss += criterion(output, y).item() * len(y)

  total_loss = total_loss/len(data_loader.dataset)
  return total_loss 


def main(args):
  train_params = args.get("train_params")
  files_ = args.get("files")
  device = torch.device(train_params.get("device"))
  model_params = args.get("model_params")


  submission_df = pd.read_csv(files_.get("data_submission"))
  train_df = pd.read_csv(files_.get("data_train"))
  test_df = pd.read_csv(files_.get("data_test"))
  X_trn, X_val = get_X(train_df,test_df)
  # y_trn = get_y(train_df,test_df)[:,np.newaxis]
  y_trn = get_y(train_df,test_df)[:,np.newaxis]
  # print('y_trn')
  # print(y_trn)

  ds = CustomDataset(X_trn.astype(np.float32), y_trn.astype(np.float32))
  ds_val = CustomDataset(X_val.astype(np.float32))

  dl_params = train_params.get("data_loader_params")
  dl = DataLoader(ds, batch_size=dl_params.get("batch_size"), shuffle=dl_params.get("shuffle"))
  dl_val = DataLoader(ds_val, batch_size=dl_params.get("batch_size"))

  model = ANN(X_trn.shape[-1] ,model_params.get("hidden_dim")).to(device)
  print(model)
  opt_params = train_params.get("optim_params")
  optimizer = torch.optim.AdamW(model.parameters(), lr=opt_params.get("lr"))
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=0.00001)

  history = {
    'loss':[],
    'val_loss':[],
    'lr':[]
  }
  
  if args.get("train"):
    pbar = range(train_params.get("epochs"))
    if train_params.get("pbar"):
      pbar = tqdm(pbar)
    
    print("Learning Start!")
    early_stopper = EarlyStopper(train_params.get("patience") ,train_params.get("min_delta"))
    for _ in pbar:
      loss = train(model, RMSELoss(), optimizer, dl, device)
      history['lr'].append(optimizer.param_groups[0]['lr'])
      scheduler.step(loss)
      history['loss'].append(loss) 
      pbar.set_postfix(trn_loss=loss)
      if early_stopper.early_stop(model, loss, files_.get("output")+files_.get("name")+'_earlystop.pth'):
        print('Early Stopper run!')            
        break
    get_graph(history, files_.get("name"))  
    print("Done!")
    torch.save(model.state_dict(), files_.get("output")+files_.get("name")+'.pth')
    
    model = ANN(X_trn.shape[-1] ,model_params.get("hidden_dim")).to(device)
    if torch.load(files_.get("output")+files_.get("name")+'_earlystop.pth'):
      model.load_state_dict(torch.load(files_.get("output")+files_.get("name")+'_earlystop.pth'))
    else:
      model.load_state_dict(torch.load(files_.get("output")+files_.get("name")+'.pth'))
    model.eval()
    
    pred = []
    with torch.inference_mode():
      for x in dl_val:
        x = x[0].to(device)
        out = model(x)
        pred.append(out.detach().cpu().numpy())
    
    submission_df['ECLO'] = np.concatenate(pred).squeeze()
    submission_df.to_csv(files_.get("submission")+files_.get("name")+'.csv',index=False)

  print('------------------------------------------------------------------')
  if args.get("validation"):
    model = ANN(X_trn.shape[-1] , model_params.get("hidden_dim")).to(device)
    scores = Validation(X_trn, y_trn, train_params.get("patience"), train_params.get("min_delta"))
    scores = pd.DataFrame(scores.kfold(model, n_splits=5, epochs=train_params.get("epochs"), lr=opt_params.get("lr"), 
                                      batch=dl_params.get("batch_size"), shuffle=True, random_state=2024))
    print(pd.concat([scores, scores.apply(['mean', 'std'])]))
    
  return


def get_args_parser(add_help=True):
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config/config.py", type=str, help="configuration file")
  # parser.add_argument("-mode", "--multi-mode", default=False, type=bool, help="multi train mode")
  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  print(args.config)
  exec(open(args.config).read())
  main(config)
  # main(args)