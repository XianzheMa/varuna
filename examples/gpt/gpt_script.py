import torchvision.transforms as transforms
import torchvision
import torch
from model.varuna_gpt import VarunaGPT
from utils.pretrain_corpus import HuggingFaceDatasetWrapper
from utils.tokenizer import build_tokenizer
from torch.utils.data.distributed import DistributedSampler

import argparse
from varuna import Varuna, get_varuna_config
import wandb
from datasets import load_dataset
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
parser.add_argument('--fp16', action='store_true', default=False, help='use fp16')
parser.add_argument('--seq_length', type=int, default=2048, metavar='N', help='-')
parser.add_argument('--embedding_dim', type=int, default=2048, metavar='N', help='-')
parser.add_argument('--num_layers', type=int, default=4, metavar='N', help='-')
parser.add_argument('--num_heads', type=int, default=16, metavar='N', help='-')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N', help='-')

args = parser.parse_args()

# if args.rank == 0:
#     wandb.login()

torch.cuda.set_device(args.local_rank)

torch.distributed.init_process_group(
    backend='gloo',
    init_method='env://',
    rank=args.rank,
)

tokenizer = build_tokenizer(args)
hf_dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train')
trainset = HuggingFaceDatasetWrapper(hf_dataset, tokenizer, args.seq_length)


def get_batch_fn(size, device=None):
    batch = next(iter(torch.utils.data.DataLoader(trainset, batch_size=size)))
    return get_dict_batch(batch, device)

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

pipeline_parallel_size, data_parallel_size = get_varuna_config(args.stage_to_rank_map)
global_batch_size = args.batch_size * data_parallel_size
model = Varuna(model, args.stage_to_rank_map, get_batch_fn, global_batch_size,
               args.chunk_size, fp16=args.fp16, device=args.local_rank)


# this seems to be the only optimizer suitable for fp16 with varuna
optimizer = apex_optimizer(model.parameters(), lr=args.lr)
model.set_optimizer(optimizer)


# varuna requires every worker to load data although intermediate ones don't need
sampler = DistributedSampler(trainset, num_replicas=data_parallel_size, rank=model.rank_within_stage)
# varuna internally splits a full mini batch to micro batches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=sampler)


model.train()
for step, batch in enumerate(trainloader):
    # important: shouldn't use optimizer.zero_grad() as varuna has special logic
    model.zero_grad()
    batch = get_dict_batch(batch, args.local_rank)
    loss, overflow, grad_norm = model.step(batch)
    if not overflow:
        optimizer.step()
    if step % 10 == 0:
        print(f"step {step} loss {loss} overflow {overflow} grad_norm {grad_norm}")


