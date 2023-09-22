import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import logging
import argparse
from varuna import Varuna, get_varuna_config, get_this_rank_config_varuna, CutPoint
from apex.optimizers import FusedLAMB as apex_optimizer
# configure basic logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)



class VarunaAlexNet(nn.Module):
    def __init__(self, num_classes, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()


        self.cutpoints = nn.ModuleList([CutPoint() for _ in range(3)])


    def forward(self, image, label):
        image = self.features(image)
        image = self.cutpoints[0](image)
        image = self.avgpool(image)
        image = self.cutpoints[1](image)
        image = torch.flatten(image, 1)
        image = self.classifier(image)
        image = self.cutpoints[2](image)
        return self.loss_fn(image, label)


parser = argparse.ArgumentParser('varuna example script')
parser.add_argument('--rank', type=int, help='rank of the current process')
parser.add_argument('--local_rank', type=int, help='local rank of the current process')
parser.add_argument('--stage_to_rank_map', default=None, type=str, help="stage to rank map of Varuna model")
parser.add_argument("--chunk_size", type=int,default=None, help="number of microbatches for pipeline")
parser.add_argument("--batch-size", type=int, default=None, help="per-process batch size given by varuna")
parser.add_argument('--fp16', action='store_true', default=False, help='use fp16')

args = parser.parse_args()

torch.cuda.set_device(args.local_rank)

torch.distributed.init_process_group(
    backend='gloo',
    init_method='env://',
    rank=args.rank,
)

transform_list = [
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
if args.fp16:
    transform_list.append(transforms.ConvertImageDtype(torch.half))

if args.rank == 0:
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose(transform_list))
    torch.distributed.barrier()
else:
    torch.distributed.barrier()
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=False,
                                                transform=transforms.Compose(transform_list))

def get_batch_fn(size, device=None):
    batch = next(iter(torch.utils.data.DataLoader(trainset, batch_size=size)))
    return get_dict_batch(batch, device)

def get_dict_batch(batch, device=None):
    if device is not None:
        batch = batch[0].to(device), batch[1].to(device)
    return {'image': batch[0], 'label': batch[1]}


model = VarunaAlexNet(num_classes=10)
pipeline_parallel_size, data_parallel_size = get_varuna_config(args.stage_to_rank_map)
global_batch_size = args.batch_size * data_parallel_size
model = Varuna(model, args.stage_to_rank_map, get_batch_fn, global_batch_size,
               args.chunk_size, fp16=args.fp16, device=args.local_rank)


# this seems to be the only optimizer suitable for fp16 with varuna
optimizer = apex_optimizer(model.parameters(), lr=0.00001)
model.set_optimizer(optimizer)


# FIXME: varuna seems to require every worker to load data although intermediate ones don't need
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


