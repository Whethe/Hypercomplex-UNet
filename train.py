import argparse
import time

import numpy as np
import yaml
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from get_instances import *
from utils.utils_functions import Logger, set_seeds, r2c, display_img
from utils.utils_devices import initialize_device


def setup(arguments):
    config_path = arguments.config
    with open(config_path, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)

    """================================================================================="""
    """================================= read configs ================================="""
    """================================================================================="""
    device_params = configs.get('device_params', {})

    train_params = configs.get('train_params')
    dataset_params = configs['dataset_params']
    model_params = configs['model_params']

    """================================================================================="""
    """================================= parse configs ================================="""
    """================================================================================="""
    # device params
    device = initialize_device(
        visible_devices=device_params.get('visible_devices', 1),
        device_id=device_params.get('device_id', 0)
    )
    
    print(device)

    # dataset params
    phases = ['train', 'val'] if dataset_params.get("val_data") else ['train']
    batch_size = dataset_params.get("batch_size", 8)
    dataset_name = dataset_params.get("dataset_name")

    # train params
    model_name = train_params.get("model_name")
    print(model_params)
    restore_weights = train_params.get("restore_weights")  # 'model', 'all', False
    restore_path = train_params.get("restore_path")
    epochs = train_params.get("epochs")

    loss_name = train_params.get("loss_name")
    score_names = train_params.get("score_names")
    optim_name = train_params.get("optim_name")
    optim_params = train_params.get('optim_params', {})
    print(f"optim_params: {optim_params}")
    scheduler_name = train_params.get('scheduler_name', None)
    scheduler_params = train_params.get('scheduler_params', {})
    print(f"scheduler_params: {scheduler_params}")

    if scheduler_params:
        for key in ['factor', 'min_lr', 'patience']:
            if key in scheduler_params:
                scheduler_params[key] = float(scheduler_params[key])

    if optim_params:
        for key in ['lr', 'weight_decay']:
            if key in optim_params:
                optim_params[key] = float(optim_params[key])

    # config info
    config_name = configs['config_name'] + '_' + datetime.now().strftime("%d%b%I%M%P") #ex) base_04Jun0243pm

    """================================================================================="""
    """================================= init training ================================="""
    """================================================================================="""

    # dirs, logger, writers, saver =========================================
    workspace = os.path.join(arguments.workspace, config_name)  # workspace/config_name
    checkpoints_dir, log_dir = get_dirs(workspace)  # workspace/config_name/checkpoints ; workspace/config_name/log.txt
    tensorboard_dir = os.path.join(arguments.tensorboard_dir, config_name)  # runs/config_name
    logger = Logger(log_dir)
    writers = get_writers(tensorboard_dir, phases)
    saver = CheckpointSaver(checkpoints_dir)

    # dataloaders, model, loss f, score f, optimizer, scheduler================================
    dataloaders = get_loaders(dataset_name, dataset_params, batch_size, phases)
    model = get_model(model_name, model_params, device)
    loss_f = get_loss(loss_name)
    score_fs = get_score_fs(score_names)
    val_score_name = score_names[0]
    optim_params['params'] = model.parameters()
    optimizer, scheduler = get_optim_scheduler(optim_name, optim_params, scheduler_name, scheduler_params)

    # load weights ==========================================
    if restore_weights:
        start_epoch, model, optimizer, scheduler = saver.load(restore_path, restore_weights, model, optimizer,
                                                              scheduler, device)
    else:
        start_epoch = 0

    # if torch.cuda.device_count()>1:
    #     model = nn.DataParallel(model)

    return (configs, device, epochs, start_epoch, phases, workspace,
    logger, writers, saver, dataloaders, model, loss_f,
    score_fs, val_score_name, optimizer, scheduler)

def train_epoch(phase, model, dataloader, device, loss_f, score_fs, optimizer, scheduler, configs):
    """Training/validation process for a single epoch"""
    running_score = defaultdict(float)
    is_train = (phase == 'train')

    if is_train:
        model.train()
    else:
        model.eval()

    for i, (x, y, mask) in enumerate(tqdm(dataloader, desc=f'{phase}')):

        x = x.to(device)  # (B, 2, H, W) - undersampled k-space
        y = y.to(device)  # (B, 2, H, W) - ground truth single-channel
        t = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        batch_size = x.shape[0]

        # Forward propagation and loss calculation
        with torch.set_grad_enabled(is_train):
            # prediction
            y_pred = model(x, t)  # (B, 2, H, W)
            # Calculate loss between y_pred and y
            loss = loss_f(y_pred, y)

        # Reverse propagation (training phase only)
        if is_train:
            optimizer.zero_grad()
            loss.backward()

            # gradient clip
            if configs.get('gradient_clip', False):
                clip_value = configs.get('clip_value', 1.0)
                nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)

            optimizer.step()

        # Record losses
        running_score['loss'] += loss.item() * batch_size

        # # Converting to the complex number field to compute metrics
        # with torch.no_grad():
        #     y_complex = r2c(y.detach().cpu().numpy(), axis=1)  # (B, 1, H, W) complex
        #     y_pred_complex = r2c(y_pred.detach().cpu().numpy(), axis=1)  # (B, 1, H, W) complex
        #
        #     # Calculating magnitude is used to evaluate
        #     y_mag = np.abs(y_complex)  # (B, H, W) float
        #     y_pred_mag = np.abs(y_pred_complex)  # (B, H, W) float
        #
        #     # Calculation of assessment indicators
        #     for score_name, score_f in score_fs.items():
        #         running_score[score_name] += score_f(y_mag, y_pred_mag) * batch_size

        with torch.no_grad():
            y = np.abs(r2c(y.detach().cpu().numpy(), axis=1))
            y_pred = np.abs(r2c(y_pred.detach().cpu().numpy(), axis=1))

            for score_name, score_f in score_fs.items():
                running_score[score_name] += score_f(y, y_pred) * y_pred.shape[0]

        # Cleaning up memory in the validation phase
        if not is_train:
            torch.cuda.empty_cache()

    # Calculation of average indicators
    epoch_score = {
        score_name: score / len(dataloader.dataset)
        for score_name, score in running_score.items()
    }

    # Learning rate scheduler stepping (training phase only)
    if is_train and scheduler:
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

    return epoch_score


def visualize_results(writer, x, y, y_pred, mask, score, epoch, phase):
    x_vis = np.abs(r2c(x.detach().cpu().numpy()))  # (B, H, W)
    y_vis = y.detach().cpu().numpy()[:, 0]  # (B, H, W)
    y_pred_vis = y_pred.detach().cpu().numpy()[:, 0]  # (B, H, W)

    fig = display_img(x_vis, mask.detach().cpu().numpy(), y_vis, y_pred_vis, score)
    writer.add_figure(f'reconstruction_{phase}', fig, epoch)


def main(arguments):
    # Settings
    (configs, device, epochs, start_epoch, phases, workspace,
     logger, writers, saver, dataloaders, model, loss_f,
     score_fs, val_score_name, optimizer, scheduler) = setup(arguments)

    # logger
    logger.write('config path: ' + arguments.config)
    logger.write('workspace: ' + workspace)
    logger.write('description: ' + configs['description'])
    logger.write('\n')
    logger.write('train start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.write('-----------------------')

    # set seed
    if arguments.seed:
        set_seeds(arguments.seed)

    start = time.time()

    # training loop
    for epoch in range(start_epoch, epochs):
        for phase in phases:
            epoch_score = train_epoch(
                phase=phase,
                model=model,
                dataloader=dataloaders[phase],
                device=device,
                loss_f=loss_f,
                score_fs=score_fs,
                optimizer=optimizer,
                scheduler=scheduler,
                configs=configs
            )

            # if 'val' in phases:
            #     # scheduler.step
            #     if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            #         scheduler.step(val_score[val_score_name])

            # record
            for score_name, score in epoch_score.items():
                writers[phase].add_scalar(score_name, score, epoch)

            if phase == 'val' and scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_score[val_score_name])

            # record learning rate
            if arguments.write_lr:
                writers[phase].add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            # visualize result
            if arguments.write_image > 0 and (epoch % arguments.write_image == 0):
                with torch.no_grad():
                    x, y, mask = next(iter(dataloaders[phase]))
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    visualize_results(
                        writers[phase], x[-1], y[-1], y_pred[-1],
                        mask[-1], epoch_score[val_score_name], epoch, phase
                    )

            # record lambda
            if arguments.write_lambda and hasattr(model, 'lam'):
                print('lam:', model.lam.item())
                writers[phase].add_scalar('lambda', model.lam.item(), epoch)

            # logger.write(
            #     'epoch {}/{} {} score: {:.4f}\tloss: {:.4f}'.format(
            #         epoch, epochs, phase, epoch_score[val_score_name],
            #         epoch_score['loss'])
            # )

            logger.write(
                'epoch {}/{} {} Score: {:.2f}\tloss: {:.4f}'.format(
                    epoch, epochs, phase,
                    epoch_score['PSNR'],
                    epoch_score['loss']
                )
            )

            # save model
            if phase == 'val':
                saver.save_model(model, epoch_score[val_score_name], epoch, final=False)

        if epoch % arguments.save_step == 0:
            saver.save_checkpoints(epoch, model, optimizer, scheduler)

    if phase == 'train':
        saver.save_model(model, epoch_score[val_score_name], epoch, final=True)

    for phase in phases:
        writers[phase].close()

    logger.write('-----------------------')
    logger.write('total train time: {:.2f} min'.format((time.time() - start) / 60))
    logger.write('best score: {:.4f}'.format(saver.best_score))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rain MRI Reconstruction Model")

    parser.add_argument(
        "--config",
        type=str, required=False,
        default="configs/train_unet.yaml",
        help="config file path"
        )

    parser.add_argument("--workspace", type=str, default='./workspace')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs')
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--write_lr", type=bool, default=True)
    parser.add_argument("--write_image", type=int, default=0)
    parser.add_argument("--write_lambda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    main(args)
