import os
import torch
import torch.distributed as dist
import argparse


def init_ddp():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1)
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank

    # 新增3：DDP backend初始化
    #   a.根据local_rank来设定当前使用哪块GPU
    torch.cuda.set_device(local_rank)
    #   b.初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
    dist.init_process_group(backend='nccl')

    # 新增4：定义并把模型放置到单独的GPU上，需要在调用`model=DDP(model)`前做哦。
    #       如果要加载模型，也必须在这里做哦。
    return local_rank


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    from logging import Logger
    logger_info = Logger.info
    def info(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            logger_info(*args, **kwargs)
    Logger.info = info
    
    
    


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode():
    args = argparse.ArgumentParser()
    args.dist_url = "env://"
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,  world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


def gather_object_across_processes(object):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to list for consistency with the distributed case.
        return list(object)
    output = [None] * dist.get_world_size()
    dist.barrier()
    dist.all_gather_object(output, object)
    return output