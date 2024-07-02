import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import os
import datetime
import random
import cv2
from tensorboardX import SummaryWriter
from models import UNet  
import utils  
import data_augmentation_outline as data_augmentation 
import torch.nn.functional as F
#implement early stopping criteria


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
    #y_pred = torch.softmax(y_pred, dim=1)
    if ignore_background:
        y_true = y_true[:, 1:, :, :]
        y_pred = y_pred[:, 1:, :, :]
    axes = (0, 2, 3)
    eps = 1e-7
    num = (2 * torch.sum(y_true * y_pred, axis=axes) + eps)
    denom = torch.sum(y_true, axis=axes) + torch.sum(y_pred, axis=axes) + eps
    score = num / denom
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
    if delta_abs < min_delta:
        print('Stopping early...')
        return False
    else:
        return False
        


def get_batch(images, labels, labels_bcs, params, mode,device):
    ct_batch = np.zeros(shape=[params.dict['batch_size'],
                               params.dict['patch_shape'][0],
                               params.dict['patch_shape'][1],
                               params.dict['patch_shape'][2]])

    gt_batch = np.zeros(shape=[params.dict['batch_size'],
                               params.dict['patch_shape'][0],
                               params.dict['patch_shape'][1],
                               params.dict['patch_shape'][2]])

    for patch in range(0, params.dict['batch_size']):
        if mode == 'Train':
            # Select specific subset to sample from more often.
            # Only for our specific version of the dataset.
            if random.randint(0, 100) > 50:
                patient = random.choice([61, 176, 276, 427, 556, 665, 783,
                                         1069, 1127, 1326, 2335, 2618])
            else:
                patient = random.randint(0, images.shape[-1] - 1)
            ct = images[:, :, patient]
            gt = labels[:, :, patient]
            gt_bcs = labels_bcs[:, :, patient]

        else:
            patient = random.randint(0, images.shape[-1] - 1)
            ct = images[:, :, patient]
            gt = labels[:, :, patient]
            gt_bcs = labels_bcs[:, :, patient]

        print(f"Selected patient: {patient}")
        #print(f"CT shape: {ct.shape}, GT shape: {gt.shape}")

        # Create contour from ground truth
        contours2, hierarchy2 = cv2.findContours(gt.astype(np.uint8),
                                                 cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
        print(f"Number of contours found: {len(contours2)}")
        img_contours2 = np.zeros(gt.shape)
        for i in range(0, len(contours2)):
            if (cv2.contourArea(contours2[i]) > 3000):
                cv2.drawContours(img_contours2,
                                 contours2,
                                 i,
                                 (255, 255, 255),
                                 1,
                                 8,
                                 hierarchy2)

        # Fill contour
        gt = np.copy(img_contours2)
        _ = cv2.drawContours(gt,
                             [max(contours2, key=cv2.contourArea)],
                             -1,
                             1,
                             thickness=-1)

        # Perform augmentations
        num_augments = np.random.randint(1, params.dict['number_of_augmentations'] + 1)
        ct, gt, _ = data_augmentation.apply_augmentations(ct,
                                                          gt,
                                                          gt_bcs,
                                                          num_augments)
        ct_batch[patch, :, :, 0] = ct
        gt_batch[patch, :, :, 0] = gt
    

    ct_batch = np.transpose(ct_batch, (0, 3, 1, 2))
    gt_batch = np.transpose(gt_batch, (0, 3, 1, 2))

    # Checking if the channel dimension is singleton
    if gt_batch.shape[1] == 1:  
        gt_batch = np.squeeze(gt_batch, axis=1)
    ct_batch = torch.tensor(ct_batch, dtype=torch.float32).to(device)
    gt_batch = torch.tensor(gt_batch, dtype=torch.long).to(device)
    gt_batch = F.one_hot(gt_batch, num_classes=params.dict['num_classes_bc']).permute(0, 3, 1, 2).float().to(device)
    return ct_batch, gt_batch


def train(model, device, optimizer, data, labels, loss_function, regularization_factor=0.0001,step=0):
    model.train()
    optimizer.zero_grad()
    outputs = model(data)
    outputs_softmax = F.softmax(outputs, dim=1) 
    loss = loss_function(outputs_softmax, labels)  
    # Adding L2 Regularization 
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    total_loss = loss + regularization_factor * l2_reg
    total_loss.backward()
    optimizer.step()    
    return outputs_softmax, total_loss.item()  

def validate(model, device, data, labels, loss_function, regularization_factor = 0.0001,step=0):
    model.eval()
    total_loss = 0.0
    l2_reg = torch.tensor(0.).to(device)
    with torch.no_grad():
        outputs = model(data)
        outputs_softmax = F.softmax(outputs, dim=1)  
        loss = loss_function(outputs_softmax, labels)  
        for param in model.parameters():
            l2_reg += torch.norm(param)
        total_loss = loss + regularization_factor * l2_reg
    return outputs_softmax, total_loss.item()  


def main():
    # Load parameters
    param_path = os.getcwd() + '/params.json'
    params = utils.Params(param_path)


    # Load training and validation data
    images_t, labels_t, labels_bcs_t, names_t = utils.load_dataset_contour_creation(params.dict['data_path_train'], params)
    images_v, labels_v, labels_bcs_v, names_v = utils.load_dataset_contour_creation(params.dict['data_path_val'], params)
    
    print('Training set size: ', np.shape(images_t)[2])
    print('Validation set size: ', np.shape(images_v)[2])
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Define model
    model = UNet(params,params.dict['num_classes_bc']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.dict['learning_rate'])
    
    scheduler = ExponentialLR(optimizer, gamma=params.dict['decay_rate'])
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # Define loss function
    loss_function = dice_loss2
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

    # Add model graph to TensorBoard
    ct_batch, gt_batch = get_batch(images_t, labels_t, labels_bcs_t, params, mode='Train', device=device)
    train_writer.add_graph(model, ct_batch)

    best_val_loss = float('inf')
    for step in range(params.dict['num_steps'] + 1):
        
        ct_batch, gt_batch = get_batch(images_t, labels_t, labels_bcs_t, params, mode='Train',device=device) 
        train_outputs, train_loss = train(model, device, optimizer, ct_batch, gt_batch, loss_function, regularization_factor=0.0001,step=step)
        
        train_dice = dice_score(gt_batch, train_outputs).item()        
        train_writer.add_scalar('Train Loss', train_loss, step)        
        train_writer.add_scalar('Train Dice', train_dice, step)

        # Log learning rate
        for param_group in optimizer.param_groups:
            train_writer.add_scalar('Learning Rate', param_group['lr'], step)

        # Log gradients and weights
        #for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        train_writer.add_histogram(f'{name}.grad', param.grad, step)
        #    train_writer.add_histogram(name, param, step)
        
        if step % params.dict['train_eval_step'] == 0:
            print(f"Iteration {step}, Loss: {train_loss:.5f}, Dice: {train_dice:.5f}")    
            
        if step % params.dict['val_eval_step'] == 0:
            ct_batch_val, gt_batch_val = get_batch(images_v, labels_v, labels_bcs_v, params, mode='Val',device=device)
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