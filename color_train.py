import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from color_model import _NetG
import color_dataset
from color_dataset import DatasetFromHdf5
import glob, math, time
import numpy as np
import scipy.io as sio
from scipy.io.matlab.mio import loadmat
import h5py
from torchsummary import summary

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DIDN Train")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.0001")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 0")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def main():
    global opt, model
    opt = parser.parse_args()
    opt.gpus = '0'
    print(opt)
    opt.cuda = True

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5("./data/training_RGB_5to50_uint8_samples.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=True)

    print("===> Building model")
    model = _NetG()
    criterion = nn.L1Loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    summary(model, (3, 64, 64))

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))

            checkpoint = torch.load(opt.resume, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint['model'].state_dict())
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    max_psnr = 0
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        max_psnr = train(training_data_loader, optimizer, model, criterion, epoch, max_psnr)
        save_checkpoint(model, epoch, 'end', 'end_ep')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 2 every 3 epochs"""
    lr = optimizer.param_groups[0]["lr"]
    if epoch % 3 == 1:
        if epoch > 1:
            lr = optimizer.param_groups[0]["lr"] / 2

    return lr


def train(training_data_loader, optimizer, model, criterion, epoch, max_psnr):

    lr = adjust_learning_rate(optimizer, epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 0):
        batch = color_dataset.tensor_augmentation(batch)  # data augmentation (random rotation / flip)
        input, target = Variable(batch[0] / 255.), Variable(batch[1] / 255., requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 4000 == 0:  # calculate validation PSNR every 4000 iteration.
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))
            with torch.no_grad():
                origin_list = glob.glob("./data/Val/Set5/original_mat_int/" + "*.*")
                noisy_list50 = glob.glob("./data/Val/Set5/noisy_mat_s50_int/" + "*.*")
                noisy_list30 = glob.glob("./data/Val/Set5/noisy_mat_s30_int/" + "*.*")
                noisy_list10 = glob.glob("./data/Val/Set5/noisy_mat_s10_int/" + "*.*")
                model.eval()
                avg_psnr_predicted = [0, 0, 0]
                avg_psnr_noisy = 0.0
                ct = 0.0

                for n in range(origin_list.__len__()):
                    origin_name = origin_list[n]
                    noisy_name50 = noisy_list50[n]
                    noisy_name30 = noisy_list30[n]
                    noisy_name10 = noisy_list10[n]
                    origin = sio.loadmat(origin_name)['origin']/255.
                    noisy_ = []
                    for i in range(3):
                        noisy_.append(np.zeros((origin.shape[2], origin.shape[0], origin.shape[1])))
                    out_ = np.zeros(origin.shape)
                    noisy = []
                    noisy.append(sio.loadmat(noisy_name50)['noisy']/255.)
                    noisy.append(sio.loadmat(noisy_name30)['noisy'] / 255.)
                    noisy.append(sio.loadmat(noisy_name10)['noisy'] / 255.)

                    origin = origin.astype(float)
                    psnr_noisy = output_psnr_mse(origin, noisy[0])
                    avg_psnr_noisy += psnr_noisy

                    for n in range(3):
                        for k in range(3):
                            noisy_[n][k, :, :] = noisy[n][: ,:, k]
                        noisy_[n] = Variable(torch.from_numpy(noisy_[n]).float()).view(1, 3, noisy_[n].shape[1], noisy_[n].shape[2])
                        if opt.cuda:
                            noisy_[n] = noisy_[n].cuda()

                        out = model(noisy_[n])
                        out = out.cpu()
                        out = out.data[0].numpy().astype(np.float32)

                        out[out < 0] = 0
                        out[out > 1] = 1
                        for k in range(3):
                            out_[:, :, k] = out[k, :, :]  # 256, 256, 3

                        psnr_predicted = output_psnr_mse(origin, out_)
                        avg_psnr_predicted[n] += psnr_predicted
                    ct += 1
                for n in range(3):
                    avg_psnr_predicted[n] = avg_psnr_predicted[n] / ct
                avg_psnr_noisy = avg_psnr_noisy / ct
                if iteration == 0:
                    print("PSNR_noisy=", avg_psnr_noisy)
                print("PSNR_predicted_s50=", avg_psnr_predicted[0])
                print("PSNR_predicted_s30=", avg_psnr_predicted[1])
                print("PSNR_predicted_s10=", avg_psnr_predicted[2])
                avg_psnr_avg = (avg_psnr_predicted[0]+avg_psnr_predicted[1]+avg_psnr_predicted[2])/3
                if iteration == 0 and epoch == 1:
                    max_psnr = avg_psnr_avg
                    psnr_name = "%0.2f" % avg_psnr_avg
                    save_checkpoint(model, epoch, iteration, psnr_name)
                else:
                    if max_psnr < avg_psnr_avg:
                        max_psnr = avg_psnr_avg
                        psnr_name = "%0.2f" % avg_psnr_avg
                        save_checkpoint(model, epoch, iteration, psnr_name)

            model.train()

    return max_psnr


def save_checkpoint(model, epoch,iteration, psnr_name):
    model_out_path = "checkpoint/" + "model_{}db_".format(psnr_name) + "{}ep_".format(epoch) + "{}it_.pth".format(iteration)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


if __name__ == "__main__":
    main()
