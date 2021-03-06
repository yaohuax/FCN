import argparse
import os
import os.path as osp
import torch
from train_fcn32s import get_parameters
from fcn.datasets.mli import ImageList
from fcn import Trainer
import torchvision.transforms as transforms
from fcn.models import FCN16s
from fcn.models import FCN32s
from fcn.models import FCN8s

"could add more"
configurations = {
    1: dict(
        max_iteration=100000,
        lr=1.0e-12,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
        fcn32s_pretrained_model='input directory of fcn16 model in here',
    )
}

def main():
    cfg = configurations
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
    train_dataloader = torch.utils.data.DataLoader(
        ImageList(fileList="/home/yaohuaxu1/FCN/train.txt",
                  transform=transforms.Compose([
                      transforms.ToTensor(), ])),
        shuffle=False,
        num_workers=8,
        batch_size=1)
    model = FCN8s()
    start_epoch = 0
    start_iteration = 0
    fcn16s = FCN16s()
    fcn16s.load_state_dict(torch.load(cfg[1]['fcn16s_pretrained_model']))
    model.copy_params_from_fcn32s(fcn16s)
    if cuda:
        model = model.cuda()
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg[1]['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg[1]['lr'],
        momentum=cfg[1]['momentum'],
        weight_decay=cfg[1]['weight_decay'])

    trainer = Trainer(cuda=False,
                      model=model,
                      optimizer=optim,
                      train_loader=train_dataloader,
                      val_loader=train_dataloader,
                      max_iter=cfg[1]['max_iteration'],
                      size_average=False
                      )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()