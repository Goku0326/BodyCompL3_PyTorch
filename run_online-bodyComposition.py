import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import utils
import os
import datetime
import random
from torch.optim.lr_scheduler import ExponentialLR
from models import UNet  
from shutil import copyfile
from tensorboardX import SummaryWriter



def run_once(f):
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, 'has_run'):
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

def dice_score(y_true, y_pred, ignore_background=True):
    if ignore_background:
    #to exclude the first channel - representing the background
        y_true = y_true[:, 1:, :, :]
        y_pred = y_pred[:, 1:, :, :]
    axes= (0, 2, 3)    
    eps = 1e-7
    num = (2*torch.sum(y_true * y_pred, axis = axes)+eps)
    denom = torch.sum(y_true, axis=axes) + torch.sum(y_pred, axis=axes)+ eps
    score = num/denom
    return score.mean()


def dice_loss2(y_true, y_pred, ignore_background=False):
    if ignore_background:
    #to exclude the first channel - representing the background
        y_true = y_true[:, 1:, :, :]
        y_pred = y_pred[:, 1:, :, :]
    axes= (0, 2, 3)    
    eps = 1e-7
    num = (2*torch.sum(y_true * y_pred, axis = axes)+eps)
    denom = torch.sum(y_true, axis=axes) + torch.sum(y_pred, axis=axes)+ eps
    score = num/denom
    loss = 1 - score.mean()
    return loss

def early_stopping(loss_list, min_delta=0.005, patience=20):
    """

    Parameters
    ----------
    loss_list : list
        List containing loss values for every evaluation.
    min_delta : float
        Float serving as minimum difference between loss values before
        early stopping is considered.
    patience : int
        Training will not be stopped before int(patience) number of evaluations
        have taken place.

    Returns
    -------

    """
    # TODO: Changed to list(loss_list)
    if len(list(loss_list)) // patience < 2:
        return False

    mean_previous = np.mean(loss_list[::-1][patience:2 * patience])
    mean_recent = np.mean(loss_list[::-1][:patience])
    delta_abs = np.abs(mean_recent - mean_previous)  # abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    return False
    if delta_abs < min_delta:
        print('Stopping early...')
        return False
    else:
        return False
        

def get_batch(images, labels, params, device):
    # Prepare batch tensors directly on the correct device with the correct dimensions
    ct_batch = torch.zeros([params.dict['batch_size'], 1, params.dict['patch_shape'][0], params.dict['patch_shape'][1]], dtype=torch.float32, device=device)
    gt_batch = torch.zeros([params.dict['batch_size'], 1, params.dict['patch_shape'][0], params.dict['patch_shape'][1]], dtype=torch.long, device=device)

    for patch in range(params.dict['batch_size']):
        patient = random.randint(0, images.shape[0] - 1)
        ct = images[patient, 0, :, :].to(device)
        gt = labels[patient, 0, :, :].to(device)

        ct_batch[patch, 0, :, :] = ct
        gt_batch[patch, 0, :, :] = gt

    # Squeeze out the unnecessary channel dimension for one-hot encoding
    gt_batch = gt_batch.squeeze(1)


    unique_labels = torch.unique(gt_batch)
    #print(f"Unique labels in current batch before range check: {unique_labels}")

    if torch.any(gt_batch >= params.dict['num_classes']) or torch.any(gt_batch < 0):

        print(f"Label values out of expected range! Found labels: {unique_labels}")
        raise ValueError("Label values out of expected range!")

    # Apply one-hot encoding
    gt_batch = F.one_hot(gt_batch, num_classes=params.dict['num_classes']).float()
    gt_batch = gt_batch.permute(0, 3, 1, 2)  # Change shape to [batch_size, num_classes, height, width]

    return ct_batch, gt_batch




def train(model, device, optimizer, data, labels, loss_function, regularization_factor=0.0001,step=0, writer=None):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    outputs_softmax = F.softmax(outputs, dim=1) 
    loss = loss_function(outputs, labels)  
    # Adding L2 Regularization 
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    total_loss = loss + regularization_factor * l2_reg
    total_loss.backward()
    optimizer.step()    
    return outputs_softmax, total_loss.item()  

def validate(model, device, data, labels, loss_function, regularization_factor = 0.0001,step=0, writer=None):
    model.eval()
    total_loss = 0.0
    l2_reg = torch.tensor(0.).to(device)
    with torch.no_grad():
        outputs = model(data)
        outputs_softmax = F.softmax(outputs, dim=1)  
        loss = loss_function(outputs, labels)  
        for param in model.parameters():
            l2_reg += torch.norm(param)
        total_loss = loss + regularization_factor * l2_reg
    return outputs_softmax, total_loss.item()  

def main():
    # Load parameters
    param_path = os.getcwd() + '/params.json'
    params = utils.Params(param_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load training and validation data
    images_t, labels_t, names_t = utils.load_dataset_contour_mapping(params.dict['data_path_train'], params, norm=True) #[512,512,4645]
    images_v, labels_v, names_v = utils.load_dataset_contour_mapping(params.dict['data_path_val'], params, norm=True)

   
    print("Unique labels in training set:", np.unique(labels_t))
    print("Unique labels in validation set:", np.unique(labels_v))

    images_t = torch.from_numpy(images_t).float().permute(2, 0, 1).unsqueeze(1)  # [4645, 1, 512, 512]
    labels_t = torch.from_numpy(labels_t).long().permute(2, 0, 1).unsqueeze(1) 

    images_v = torch.from_numpy(images_v).float().to(device).permute(2, 0, 1).unsqueeze(1)  # Reshape if necessary
    labels_v = torch.from_numpy(labels_v).long().to(device).permute(2, 0, 1).unsqueeze(1)  

    print('Training set size: ', np.shape(images_t)[2])
    print('Validation set size: ', np.shape(images_v)[2])
    
    
    # Define model
    model = UNet(params,params.dict['num_classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.dict['learning_rate'])
    
    scheduler = ExponentialLR(optimizer, gamma=params.dict['decay_rate'])
    #criterion = nn.CrossEntropyLoss()  # or custom dice loss function
    # Define loss function
    
    loss_function = nn.CrossEntropyLoss()
    loss_list = []
    
    # Create directories for logs and models
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join(params.dict['log_path'], 'gradient_tape', current_time)
    train_log_dir = os.path.join(base_dir, 'train')
    val_log_dir = os.path.join(base_dir, 'val')
    saved_model_path = os.path.join(base_dir, 'saved_models')
    saved_weights_path = os.path.join(base_dir, 'saved_weights')
    
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(val_log_dir, exist_ok=True)
    os.makedirs(saved_model_path, exist_ok=True)
    os.makedirs(saved_weights_path, exist_ok=True)

    #tensorboard-wip
    train_writer = SummaryWriter(train_log_dir)
    val_writer = SummaryWriter(val_log_dir)

    best_val_loss = float('inf')

    # Add model graph to TensorBoard
    ct_batch, gt_batch = get_batch(images_t, labels_t, params, device=device)
    train_writer.add_graph(model, ct_batch)

    for step in range(params.dict['num_steps'] + 1):
        
        ct_batch, gt_batch = get_batch(images_t, labels_t, params,device=device) ##       
        train_outputs, train_loss = train(model, device, optimizer, ct_batch, gt_batch, loss_function, regularization_factor=0.0001,step=step)
        
        train_dice = dice_score(gt_batch, train_outputs).item()
        train_writer.add_scalar('Loss/train', train_loss, step)
        train_writer.add_scalar('Dice/train', train_dice, step)

        for param_group in optimizer.param_groups:
            train_writer.add_scalar('Learning Rate', param_group['lr'], step) 
        
        if step % params.dict['train_eval_step'] == 0:
            print(f"Iteration {step}, Loss: {train_loss:.5f}, Dice: {train_dice:.5f}")    
            
        if step % params.dict['val_eval_step'] == 0:
            ct_batch_val, gt_batch_val = get_batch(images_v, labels_v, params,device=device)
            val_outputs, val_loss = validate(model, device, ct_batch_val, gt_batch_val, loss_function, regularization_factor=0.0001,step=step)
            val_dice = dice_score(gt_batch_val, val_outputs)

            val_writer.add_scalar('Validation Loss', val_loss, step)
            val_writer.add_scalar('Validation Dice', val_dice, step)
            print(f"Validation - Iteration {step}, Loss: {val_loss:.5f}, Dice: {val_dice:.5f}")  
            scheduler.step(val_loss)
            loss_list.append(val_loss) 

            # Check for early stopping
            early_stop = early_stopping(loss_list, min_delta=0.001, patience=10)
            #if early_stop:
            #    print("Early stopping signal received at iteration = %d/%d" % (step, params.dict['num_steps']))
            #    print("Terminating training ")
#
            #    model_weights_save_path = os.path.join(saved_weights_path, f'model_weights_{step}.pth')
            #    torch.save(model.state_dict(), model_weights_save_path)
            #    print(f"Saved model weights at iteration {step} to {model_weights_save_path}")
#
#
            #    model_full_save_path = os.path.join(saved_model_path, f'model_full_{step}.pth')
            #    torch.save(model, model_full_save_path)
            #    print(f"Saved full model at iteration {step} to {model_full_save_path}")
            #    break
                        


        #saving the model
        if step % params.dict['save_model_step'] == 0 and not early_stop:
        
            model_weights_save_path = os.path.join(saved_weights_path, f'model_weights_{step}.pth')
            torch.save(model.state_dict(), model_weights_save_path)
            print(f"Saved model weights at iteration {step} to {model_weights_save_path}")


            model_full_save_path = os.path.join(saved_model_path, f'model_full_{step}.pth')
            torch.save(model, model_full_save_path)
            print(f"Saved full model at iteration {step} to {model_full_save_path}")
        
        #scheduler.step() 
    
    train_writer.close()
    val_writer.close()

if __name__ == "__main__":
    main()