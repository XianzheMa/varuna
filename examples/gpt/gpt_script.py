import os
import torch
import torch.distributed as dist
from model.varuna_gpt import VarunaGPT
from utils.pretrain_corpus import HuggingFaceDatasetWrapper
from utils.tokenizer import build_tokenizer
from utils.repeating_loader import RepeatingLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
import time
from varuna.bucket_client import BucketClient
from varuna import Varuna, get_varuna_config, Profiler
from datasets import load_dataset
import torch.distributed as dist
import logging
import sys
import wandb
import itertools

# FIXME: this seems to be the only optimizer suitable for fp16 with varuna; needs investigation
from apex.optimizers import FusedLAMB as apex_optimizer


parser = argparse.ArgumentParser('varuna example script')
# args passed by varuna launcher
parser.add_argument('--rank', type=int, help='rank of the current process')
parser.add_argument('--local_rank', type=int, help='local rank of the current process')
parser.add_argument('--stage_to_rank_map', default=None, type=str, help="stage to rank map of Varuna model")
parser.add_argument("--chunk_size", type=int,default=None, help="number of microbatches for pipeline")
parser.add_argument("--batch-size", type=int, default=None, help="per-process batch size given by varuna")

# gpt specific args
parser.add_argument('--tokenizer_type', type=str, default='BertWordPieceLowerCase', metavar='S',
                        help='which tokenizer to use.')
parser.add_argument('--vocab_file', type=str, default='./utils/bert-large-cased-vocab.txt',
                    metavar='S',
                    help='which tokenizer to use.')
parser.add_argument('--vocab_extra_ids', type=int, default=0, metavar='N', help='-')
parser.add_argument('--make_vocab_size_divisible_by', type=int, default=128, metavar='N', help='-')
parser.add_argument('--fp16', action='store_true', help='use fp16')
parser.add_argument('--seq_length', type=int, default=2048, metavar='N', help='-')
parser.add_argument('--embedding_dim', type=int, default=2048, metavar='N', help='-')
parser.add_argument('--num_layers', type=int, default=4, metavar='N', help='-')
parser.add_argument('--num_heads', type=int, default=16, metavar='N', help='-')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N', help='-')
parser.add_argument('--profiling', action='store_true', help='enable profiling')
parser.add_argument('--profile_path', type=str, default='./saved_profiles', help='path to save profiling information')
parser.add_argument('--hf_dataset_cache_path', type=str, default=None, help='path to save hf dataset cache')
parser.add_argument('--save_every', type=int, required=False, help='save model every n steps; omitted to disable saving')
parser.add_argument('--log_every', type=int, default=10, help='log every n steps')
parser.add_argument('--remote_ckpt_path', type=str, default='varuna/gpt', help='remote checkpoint path')
parser.add_argument('--local_ckpt_path', type=str, default='temp_ckpt', help='temporary local checkpoint path')
parser.add_argument('--num_iters', type=int, default=0, help='number of iterations to train in total; 0 to run infinitely')
parser.add_argument('--times', type=int, default=0, help='the n-th time to run the script')
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(lineno)d: %(message)s',
    filename=f'local_{args.times}.log',
)

torch.cuda.set_device(args.local_rank)
logging.info('begin to init')
dist.init_process_group(
    backend='gloo',
    init_method='env://',
    rank=args.rank,
)
logging.info('finish init')
tokenizer = build_tokenizer(args)
hf_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train', cache_dir=args.hf_dataset_cache_path)
trainset = HuggingFaceDatasetWrapper(hf_dataset, tokenizer, args.seq_length)


def get_dict_batch(batch, device=None):
    if device is not None:
        batch = batch[0].to(device), batch[1].to(device)
    return {'input_ids': batch[0], 'targets': batch[1]}


model = VarunaGPT(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=args.embedding_dim,
    seq_length=args.seq_length,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
)

def get_batch_fn(size, device=None):
    batch = next(iter(torch.utils.data.DataLoader(trainset, batch_size=size)))
    return get_dict_batch(batch, device)


profiler = None
if args.profiling:
    os.makedirs(args.profile_path, exist_ok=True)
    profiler = Profiler(
        model, get_batch_fn, fp16=args.fp16, device=args.local_rank,
        from_cache=True, out_folder=args.profile_path, add_to_existing=True
    )
else:
    pipeline_parallel_size, data_parallel_size = get_varuna_config(args.stage_to_rank_map)
    global_batch_size = args.batch_size * data_parallel_size
    model = Varuna(model, args.stage_to_rank_map, get_batch_fn, global_batch_size,
                   args.chunk_size, fp16=args.fp16, device=args.local_rank, local_rank=args.local_rank)


# this seems to be the only optimizer suitable for fp16 with varuna
optimizer = apex_optimizer(model.parameters(), lr=args.lr)
if args.profiling:
    remote_profile_dir = args.remote_ckpt_path + '/profiles'
    profiler.set_optimizer(optimizer)
    profiler.profile_all(list(range(1, 25)))
    # each worker has a partial profile, so we need to sync them to get the full profile
    client = BucketClient(bucket_name='torchelastic')
    client.upload_profiles('saved_profiles', remote_profile_dir=remote_profile_dir)
    # wait for all workers to finish uploading
    dist.barrier()
    client.download_profiles('saved_profiles', remote_profile_dir=remote_profile_dir)
    exit(0)

model.set_optimizer(optimizer)

pipeline_parallel_size, data_parallel_size = get_varuna_config(args.stage_to_rank_map)
if args.rank == 0:
    wandb.login()
    dict_args = vars(args)
    dict_args['pipeline_parallel_size'] = pipeline_parallel_size
    dict_args['data_parallel_size'] = data_parallel_size
    wandb.init(
        project='varuna',
        config=dict_args,
        name=f'varuna_{args.times}',
    )

logging.info(f'pipeline parallel size {pipeline_parallel_size}, data parallel size {data_parallel_size}, chunk size {args.chunk_size}')
# varuna requires every worker to load data although intermediate ones don't need
sampler = DistributedSampler(trainset, num_replicas=data_parallel_size, rank=model.rank_within_stage)
# varuna internally splits a full mini batch to micro batches
# because sailor also has 2 data loading workers
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=sampler, num_workers=2)
trainloader = RepeatingLoader(trainloader)

model.train()
running_loss = 0.0
iter_time = 0.0
log_interval = args.log_every

last_ckpt_future = None
os.makedirs(args.local_ckpt_path, exist_ok=True)

start_from = 0
client = BucketClient(bucket_name='torchelastic')
remote_tag_file_path = os.path.join(args.remote_ckpt_path, 'latest')
if client.bucket.get_blob(remote_tag_file_path) is not None:
    start_from = int(client.read_from_remote(remote_tag_file_path))
    logging.info(f'found checkpoint tag file {remote_tag_file_path} with iteration {start_from}')

if start_from > 0:
    logging.info(f'loading checkpoint from {start_from}')
    model.load_checkpoint(
        global_store=args.remote_ckpt_path,
        iteration=start_from,
        check_complete=False # by virtue of the latest tag we ensure that every checkpoint is complete
    )
else:
    logging.info('no checkpoint found, start from scratch')

logging.info(f'start from step {start_from}')

if args.num_iters <= 0:
    step_generator = itertools.count(start=start_from, step=1)
    logging.info("training for an infinite loop")
else:
    step_generator = range(start_from, args.num_iters)

for step in step_generator:
    batch = next(trainloader)
    # important: shouldn't use optimizer.zero_grad() as varuna has special logic
    start = time.time()
    model.zero_grad()
    batch = get_dict_batch(batch, args.local_rank)
    loss, overflow, grad_norm = model.step(batch)
    end = time.time()
    iter_time += end - start
    running_loss += loss
    if not overflow:
        optimizer.step()
    if step % log_interval == log_interval - 1:
        logging.info(f'step {step + 1} loss {running_loss / log_interval}; iter time {iter_time / log_interval}')
        if args.rank == 0:
            wandb.log({
                'loss': running_loss / log_interval,
                'per_iter_time': iter_time / log_interval,
            }, step=step + 1)
        running_loss = 0.0
        iter_time = 0.0

    if args.save_every is not None and step % args.save_every == args.save_every - 1:
        if last_ckpt_future is not None:
            logging.info('wait for previous ckpt upload to finish')
            start = time.time()
            error_flag = torch.zeros(1, dtype=torch.int8, device='cpu')
            try:
                last_ckpt_future.result()
            except Exception as exc:
                logging.info(f"Exception {exc} occurred during previous upload")
                error_flag[0] = 1
            finally:
                last_ckpt_future = None
                logging.info("wait for all reduce on error flag")
                dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
                logging.info("finished all reduce on error flag")
                # if error does not exist
                if error_flag.item() == 0:
                    logging.info("Previous upload succeeded")
                    if args.rank == 0:
                        # upload last tag
                        remote_tag_file_path = os.path.join(args.remote_ckpt_path, 'latest')
                        client.write_to_remote(str(model.iteration), remote_tag_file_path)
                        logging.info(f"Overwrote latest tag {step + 1}")
                else:
                    logging.info("Previous upload failed; will not overwrite latest tag")


            end = time.time()
            logging.info('waited for {:.2f} secs'.format(end - start))

        # note the ckpt number is always step + 1 because internal step is incremented already
        #  at the end of model.step(...)
        last_ckpt_future = model.checkpoint(
            global_store=args.remote_ckpt_path,
            tempdir=args.local_ckpt_path,
            shard=True,
        )
        logging.info(f'saved checkpoint at step {step + 1}')
