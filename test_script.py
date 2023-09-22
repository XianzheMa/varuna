import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import logging
from torchvision.models import AlexNet
import argparse
from varuna import Varuna, get_varuna_config, get_this_rank_config_varuna, CutPoint

# configure basic logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)



class VarunaAlexNet(AlexNet):
    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        layers = []

        num_features = len(self.features)
        self.cutpoints = nn.ModuleList([CutPoint() for _ in range(num_features - 1)])

        for i, layer in enumerate(self.features):
            layers.append(layer)
            if i < num_features-1:
                layers.append(self.cutpoints[i])

        layers.append(self.avgpool)
        layers.append(lambda x: torch.flatten(x, 1))
        layers.extend(self.classifier)

        self.varuna_layers = layers





    def forward(self, image, label):
        for layer in self.varuna_layers:
            image = layer(image)
        return self.loss_fn(image, label)


parser = argparse.ArgumentParser('varuna example script')
parser.add_argument('--rank', type=int, help='rank of the current process')
parser.add_argument('--local_rank', type=int, help='local rank of the current process')
parser.add_argument('--stage_to_rank_map', default=None, type=str, help="stage to rank map of Varuna model")
parser.add_argument("--chunk_size", type=int,default=None, help="number of microbatches for pipeline")
parser.add_argument("--batch-size", type=int, default=None, help="per-process batch size given by varuna")


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
               args.chunk_size, fp16=True, device=args.local_rank)


optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
model.set_optimizer(optimizer)


# FIXME: varuna seems to require every worker to load data although intermediate ones don't need
sampler = DistributedSampler(trainset, num_replicas=data_parallel_size, rank=model.rank_within_stage)
# varuna internally splits a full mini batch to micro batches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, sampler=sampler)


model.train()
for step, batch in enumerate(trainloader):
    optimizer.zero_grad()
    loss, overflow, grad_norm = model.step(batch)
    if not overflow:
        optimizer.step()
    if step % 10 == 0:
        print(f"step {step} loss {loss} overflow {overflow} grad_norm {grad_norm}")


