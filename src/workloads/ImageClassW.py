# This code is based on an original implementation found at https://github.com/pytorch/vision/blob/main/references/classification/train.py
# Modifications have been made by Cody Sloan and Tyson Limato
# Project: AI Benchmarking
# Last Updated: 08/22/2024

import torch
import torchvision
import time
import datetime
import os
import warnings
from tqdm import tqdm
import workloads.ImageClassificationTools.utils as utils
import psutil
import csv

def train_one_epoch(args, model, criterion, optimizer, data_loader, num_classes, device, epoch, model_ema=None, scaler=None):
    """
    Parameters
    ----------
    args : argparse.Namespace
        The arguments object that contains all of the necessary information to train an Image Classifier model. The
        allowed arguments, their types, and their purposes are defined in the get_args_parser function.
    model : torch.nn.Module
        The model to train
    criterion : A torch.nn criterion function
        The loss criterion to use for training
    optimizer : torch.optim.Optimizer
        The optimizer to use for training
    data_loader : torch.utils.data.DataLoader
        The DataLoader object that will provide the training data
    num_classes : int
        The number of classes in the dataset
    device : torch.device
        The device to use for training (cuda or cpu)
    epoch : int
        The current epoch number
    model_ema : utils.ExponentialMovingAverage
        The Exponential Moving Average model to use for training. Only used if enabled in the arguments at 
        initialization.
    scaler : torch.cuda.amp.GradScaler
        The gradient scaler to use for mixed precision training. Only used if enabled in the arguments at initialization.
    
    Description
    -----------
    Performs a single training epoch on the model. Metrics are also logged in the console during training in this function.
    
    """
    model.train()
    header = f"Epoch: [{epoch+1}/{args.epochs}]"
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=header)

    # Initialize disk I/O counters at the start of the epoch
    initial_disk_io_counters = psutil.disk_io_counters()
    initial_read_count = initial_disk_io_counters.read_count
    initial_write_count = initial_disk_io_counters.write_count

    for i, batch in progress_bar:
        start_time = time.time()  # Start time for the batch processing

        image = batch[args.image_column]
        target = batch[args.label_column]
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        maxk = num_classes if num_classes < 5 else 5
        second_acc_name = f"acc{maxk}"
        first_acc, second_acc = utils.accuracy(output, target, topk=(1, maxk))
        batch_size = image.shape[0]

        # Calculate batch time and throughput
        end_time = time.time()
        batch_time = end_time - start_time
        throughput = batch_size / batch_time

        # Update and calculate disk I/O counters
        current_disk_io_counters = psutil.disk_io_counters()
        disk_read_iops = (current_disk_io_counters.read_count - initial_read_count) / batch_time
        disk_write_iops = (current_disk_io_counters.write_count - initial_write_count) / batch_time

        # Update initial counts for the next batch
        initial_read_count = current_disk_io_counters.read_count
        initial_write_count = current_disk_io_counters.write_count

        output_path = os.path.join(args.output, "training_results.csv")
        # Log the results for each batch
        with open(output_path, 'a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                end_time, epoch, i, loss.item(), batch_time, throughput,
                disk_read_iops, disk_write_iops
            ])

        # Update the progress bar with additional info
        progress_bar.set_postfix(loss=f"{loss.item():.2f}", acc1=f"{first_acc.item():.2f}", 
                                 **{second_acc_name: f"{second_acc.item():.2f}"}, throughput=f"{throughput:.2f} img/s")


def evaluate(args, model, criterion, data_loader, num_classes, device, print_freq=100, log_suffix=""):
    """
    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate
    criterion : torch.nn.Module
        The loss criterion to use for evaluation
    data_loader : torch.utils.data.DataLoader
        The DataLoader object that will provide the evaluation data
    device : torch.device
        The device to use for evaluation (cuda or cpu)
    print_freq : int
        The frequency at which to print the evaluation metrics to the console
    log_suffix : str
        The suffix to the header of the log message. Used to differentiate between different model configurations.
    

    Description
    -----------
    Evaluates the model on the evaluation dataset provided by the data_loader. The evaluation metrics are logged in the
    console during evaluation in this function. The function returns the global average top accuracy of the model on the
    evaluation dataset.
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for batch in metric_logger.log_every(data_loader, print_freq, header):
            image = batch[args.image_column]
            target = batch[args.label_column]
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            maxk = num_classes if num_classes < 5 else 5
            second_acc_name = f"acc{maxk}"
            first_acc, second_acc = utils.accuracy(output, target, topk=(1, maxk))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(first_acc.item(), n=batch_size)
            metric_logger.meters[second_acc_name].update(second_acc.item(), n=batch_size)
            num_processed_samples += batch_size
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@{maxk} {getattr(metric_logger,second_acc_name).global_avg:.3f}")
    return 


def train_classification_model(args, train_dataloader, test_dataloader, train_sampler, num_classes):
    """
    Parameters
    ----------
    args : argparse.Namespace
        The arguments object that contains all of the necessary information to train an Image Classifier model. The
        allowed arguments, their types, and their purposes are defined in the get_args_parser function.
        This will include the model and dataset being used, and other options for the training process.
    train_dataloader : torch.utils.data.DataLoader
        The DataLoader object that will provide the training data.
    test_dataloader : torch.utils.data.DataLoader
        The DataLoader object that will provide the testing data.
    train_sampler : torch.utils.data.Sampler
        The sampler object that was used to create the training DataLoader. Used for distributed training.
    num_classes : int
        The number of classes in the dataset.
    
    Description
    -----------
    Trains an Image Classification model using the provided DataLoader objects and arguments. The training in this
    function is completely configurable using the arguments object. This function does allow for distributed training
    and mixed precision training. If the test_only argument is set to True, the model will only be tested on the test
    dataset and no training will be performed.
    """
    # Set the device to what is specified in the arguments
    device = torch.device(args.device)
    # Decide whether to make pytorch use deterministic algorithms or not
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


    # Create the model
    print("Creating Model...")
    if args.weights is not None:
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    else:
        model = torchvision.models.get_model(args.model, num_classes=num_classes)
    model.to(device)
    
    # Choose whether to synced batchnorm or not based on whether the model is distributed or not
    if args.distributed and args.sync_bn:
        utils.init_distributed_mode(args)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Create the loss criterion
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Set up the weight decay parameters
    custom_keys_weight_decay = []
    if args.bias_weight_decay:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )


    # Set up the optimizer
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")


    # Set up the gradient scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    
    # Set up the learning rate scheduler
    lr_scheduler_name = args.lr_scheduler.lower()
    if lr_scheduler_name == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif lr_scheduler_name == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif lr_scheduler_name == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )
    
    
    # Set up the warmup learning rate scheduler if that setting is configured
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    # Set up the distributed data parallel model if using distributed training
    # Need to keep a reference to the model without DDP for saving checkpoints
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    # Set up Exponential Moving Average (EMA) if it is enabled
    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
    
    
    # Load the model and optimizer from a checkpoint if resuming
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
    
    # If we only want to test the model, we do that and return without training
    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(args, model_ema, criterion, test_dataloader, num_classes, device=device, log_suffix="EMA")
        else:
            evaluate(args, model, criterion, test_dataloader, num_classes, device=device)
        return
    
    output_path = os.path.join(args.output, "training_results.csv")

    # create header for the csv file
    with open(output_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            'Timestamp (Unix)',
            'Epoch',
            'Batch',
            'Training Loss',
            'Execution Time (sec)',
            'Throughput (Seq/sec)',
            'Disk Read IOPS',
            'Disk Write IOPS'
            ])
    
    # start training loop
    print("Model Created. Start training...")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Train a single epoch and evaluate the model
        train_one_epoch(args, model, criterion, optimizer, train_dataloader, num_classes, device, epoch, model_ema, scaler)

        lr_scheduler.step()
        evaluate(args, model, criterion, test_dataloader, num_classes, device=device)
        
        # Evaluate the EMA model if it is enabled
        if model_ema:
            evaluate(args, model_ema, criterion, test_dataloader, num_classes, device=device, log_suffix="EMA")
            
        # Save the model checkpoint if the output directory is specified
        if args.output:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            # create the checkpoints directory if it doesn't exist
            
            utils.save_on_master(checkpoint, os.path.join(args.output, "checkpoints/model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output, "checkpoints/updated_checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time:  {total_time_str}")
    
