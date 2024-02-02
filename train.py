import time
import torch
from options.train_options import TrainOptions
#from datasets.two_dim_multi_stream import create_dataset
from models import create_model
from util.visualizer import Visualizer
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
#from evaluate import evaluate
import platform
from datetime import datetime
import pdb
import argparse
import numpy
import os
import sys


if platform.system() == 'Windows':
    NUM_WORKERS = 0
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    NUM_WORKERS = 4
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_net_pt as common_net
import common_pelvic_pt as common_pelvic


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    if opt.task == 'pelvic':
        num_classes = common_pelvic.NUM_CLASSES
        dataset_s = common_pelvic.Dataset(opt.dataroot, "ct", n_slices=opt.input_nc, debug=opt.debug)
        dataset_t = common_pelvic.Dataset(opt.dataroot, "cbct", n_slices=opt.input_nc, debug=opt.debug)
        _, val_data, _, val_label = common_pelvic.load_val_data(opt.dataroot)
    else:
        raise NotImplementedError(opt.task)

    opt.num_classes = num_classes

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=opt.batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=NUM_WORKERS)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=opt.batch_size, shuffle=True, pin_memory=True,
                                               drop_last=True, num_workers=NUM_WORKERS)

    dataset_size = len(dataloader_s)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    if opt.display:
        visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations
    writter = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "logs"))
    model.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    optimize_time = 0.1

    times = []
    min_seg_loss = 100
    best_dsc = 0
    patch_shape = (opt.input_nc, val_data[0].shape[1], val_data[0].shape[2])
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        if opt.display:
            visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i,(data_s, data_t) in enumerate(zip(dataloader_s, dataloader_t)):
            data = {
                "A": (data_s["image"],),
                "segA": (data_s["label"],),
                "A_paths": "",
                "B": (data_t["image"],),
                "segB": (data_s["label"],),
                "B_paths": "",
            }

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if epoch > opt.seg_start_point:  # start training seg model after CUT gets stable
                if opt.model == "cut_coseg" or opt.model == "cut_coseg_sum":
                    model.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'PCL', 'fake_S', 'real_S', 'D_S', 'GCL']
                else:
                    model.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'PCL', 'fake_S', 'real_S', 'D_S']
                model.optimize_seg_parameters()  # calculate loss functions related to seg model, update relevant weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / opt.batch_size * 0.005 + 0.995 * optimize_time

            if opt.display and total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if opt.display and total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                for name in losses.keys():
                    writter.add_scalar(name, losses[name], i + epoch*len(dataloader_s))
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                if epoch > opt.seg_start_point:
                    if losses['real_S'] < min_seg_loss:
                        model.save_networks('best')
                        min_seg_loss = losses['real_S']


            """
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            """

            iter_data_time = time.time()

        """
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            # model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        """

        # evaluation
        model.netS_B.eval()
        dsc_list = numpy.zeros((len(val_data), num_classes - 1), numpy.float32)
        with torch.no_grad():
            for i in range(len(val_data)):
                pred = common_net.produce_results(model.device, lambda x: model.netS_B(x).softmax(1).unsqueeze(2),
                                                  [patch_shape, ], [val_data[i], ], data_shape=val_data[i].shape,
                                                  patch_shape=patch_shape, is_seg=True, num_classes=num_classes)
                pred = pred.argmax(0).astype(numpy.float32)
                dsc_list[i] = common_metrics.calc_multi_dice(pred, val_label[i], num_cls=num_classes)

        model.netS_B.train()
        if dsc_list.mean() > best_dsc:
            best_dsc = dsc_list.mean()
            model.save_networks('best')

        print("%s  Epoch:%d/%d  val_dsc:%f/%f  best_dsc:%f" %
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, opt.n_epochs + opt.n_epochs_decay,
               dsc_list.mean(), dsc_list.std(), best_dsc))
        model.save_networks('last')

        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    #print("End of training, the best segmentation loss on fake B is", min_seg_loss)
    #print("=====Start evaluation====")
    #(visualizer.save_dir)

