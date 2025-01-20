import argparse
import json
from collections import defaultdict
from datetime import datetime

import time
import yaml
from PIL import Image
from tqdm import tqdm

from get_instances import *
from utils.utils_functions import *

import matplotlib.pyplot as plt

from utils.utils_devices import initialize_device


def initArgs(arguments):
    config_path = arguments.config
    with open(config_path, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)
    """================================================================================="""
    """================================= read configs ================================="""
    """================================================================================="""
    device_params = configs.get('device_params', {})

    test_params = configs.get('test_params')
    dataset_params = configs['dataset_params']
    model_params = configs['model_params']

    """================================================================================="""
    """================================= parse configs ================================="""
    """================================================================================="""
    # device params
    device = initialize_device(
        visible_devices=device_params.get('visible_devices'),
        device_id=device_params.get('device_id', 0)
    )

    print(device)

    # dataset params
    phases = ['train', 'val'] if dataset_params.get("val_data") else ['train']
    batch_size = dataset_params.get("batch_size", 8)
    dataset_name = dataset_params.get("dataset_name")

    # test params
    model_name = test_params.get("model_name")
    print(model_params)
    restore_weights = test_params.get("restore_weights")  # 'model', 'all', False
    restore_path = test_params.get("restore_path")
    score_names = test_params.get("score_names")

    # config info
    config_name = configs['config_name']

    workspace = os.path.join(arguments.workspace, config_name)  # workspace/config_name
    checkpoints_dir, log_dir = get_dirs(workspace)  # workspace/config_name/checkpoints ; workspace/config_name/log.txt

    model = get_model(model_name, model_params, device)
    score_fs = get_score_fs(score_names)

    # restore
    saver = CheckpointSaver(checkpoints_dir)
    prefix = test_params.get('restore_weights', 'final')  # Default to 'final' if not specified

    # Get list of matching checkpoint files
    matching_checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith(prefix)]

    # Handle no checkpoints found
    if not matching_checkpoints:
        # Try alternative prefixes if the preferred one is not found
        alternative_prefixes = ['best', 'final', 'inter']
        for alt_prefix in alternative_prefixes:
            matching_checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith(alt_prefix)]
            if matching_checkpoints:
                prefix = alt_prefix
                break

    # Raise error if no checkpoints found at all
    if not matching_checkpoints:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoints_dir}. Please check your checkpoint directory.")

    # Select the most recent checkpoint if multiple exist
    checkpoint_path = os.path.join(checkpoints_dir, sorted(matching_checkpoints)[-1])
    # prefix = 'best' if configs['val_data'] else 'final'
    # checkpoint_path = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.startswith(prefix)][0]

    dataloader = get_loaders(dataset_name, dataset_params, batch_size, ['test'])['test']
    model = saver.load_model(checkpoint_path, model, device)

    print(f"Model Parameters: {model_params} \n")
    print(f"Model Loaded From: {checkpoint_path}")

    return configs, device, workspace, dataloader, model, score_fs

def main(arguments):
    configs, device, workspace, dataloader, model, score_fs = initArgs(arguments)
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('test start: ' + start_time)

    new_folder_path = f"./outputImages/{start_time}' '{configs['description']}"
    os.makedirs(new_folder_path, exist_ok=True)

    start = time.time()

    running_score = defaultdict(int)

    model.eval()
    for i, (x, y, mask) in enumerate(tqdm(dataloader)):
        x = x.float().to(device)
        y = y.float().to(device)
        t = torch.zeros(x.shape[0], dtype=torch.long, device=device)

        with torch.no_grad():
            y_pred = model(x, t).float()  # (B, H, W)
        # print(f"y shape: {y.shape}, y_pred shape: {y_pred.shape}")

        y = np.abs(r2c(y.cpu().numpy(), axis=1))
        y_pred = np.abs(r2c(y_pred.cpu().numpy(), axis=1))
        # print(f"y shape: {y.shape}, y_pred shape: {y_pred.shape}")

        for score_name, score_f in score_fs.items():
            running_score[score_name] += score_f(y, y_pred) * y_pred.shape[0]

        if arguments.write_image > 0 and (i % arguments.write_image == 0):

            display_img(
                np.abs(r2c(x[-1].detach().cpu().numpy())),
                mask[-1].detach().cpu().numpy(),
                y[-1],
                y_pred[-1],
                psnr(
                    y[-1],
                    y_pred[-1]
                )
            )

            plt.savefig(os.path.join(new_folder_path, f'output{i}.png'))
            plt.close()

    epoch_score = {score_name: score / len(dataloader.dataset) for score_name, score in running_score.items()}
    for score_name, score in epoch_score.items():
        print('test {} score: {:.4f}'.format(score_name, score))

    print('-----------------------')
    print('total test time: {:.4f} sec'.format((time.time()-start)/3600))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=False, default="configs/test/test_unet.yaml",
                        help="config file path")
    parser.add_argument("--workspace", type=str, default='./workspace')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs')
    parser.add_argument("--save_step", type=int, default=10)
    parser.add_argument("--write_lr", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--write_image", type=int, default=1)
    parser.add_argument("--write_lambda", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    main(args)