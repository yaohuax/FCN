import argparse
import os
import os.path as osp
import torch
import torchfcn
from train_fcn32s import get_parameters
from fcn.datasets import mli

"could add more"
configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-12,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
        fcn32s_pretrained_model=torchfcn.models.FCN32s.download(),
    )
}


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    args = parser.parse_args()
    cfg = configurations[args.config]
 #   out = get_log_dir('fcn16s', args.config, cfg)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = osp.expanduser('')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        mli.segBase(root, split='train', transform=False),
        batch_size=1, shuffle=True, **kwargs)
    # val_loader = torch.utils.data.DataLoader(
    #     torchfcn.datasets.VOC2011ClassSeg(
    #         root, split='seg11valid', transform=True),
    #     batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN16s(n_class=2)
    start_epoch = 0
    start_iteration = 0
    fcn32s = torchfcn.models.FCN32s()
    fcn32s.load_state_dict(torch.load(cfg['fcn32s_pretrained_model']))
    model.copy_params_from_fcn32s(fcn32s)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()