import torch
import torchvision.transforms as transforms
import torch.optim as optim

from config import load_args
from preprocess import load_data
from model import Model
# from fid import InceptionV3, get_fid

import os
from PIL import Image
import imageio
import numpy as np
from tqdm import tqdm


def save_ckpt(global_step, model, optimizer):
    base_dir = "./checkpoints"
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    filename = os.path.join(base_dir, "global_step_{}.pt".format(global_step))
    torch.save({
        "step": global_step,
        "state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
    }, filename)


# reference, thank you.
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def warmup_lr(step):
    return min(step, 5000) / 5000


def _eval(gen_data, test_loader, args):
    model = InceptionV3()
    if args.cuda:
        model = model.cuda()
    model.eval()
    
    transform = transforms.Compose([
        transforms.Normalize((0.5,0.5,0.5), (.5,.5,.5)),
    ])

    real_imgs, gen_imgs = [], []
    with torch.no_grad():
        for i, (data, label) in tqdm(enumerate(test_loader)):
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            real_out = model(data)[0].view(data.size(0), -1)
            gen_out = model(transform(gen_data[i * data.size(0):(i + 1) * data.size(0)].float()))[0].view(data.size(0), -1)

            real_imgs.append(real_out.cpu().data.numpy())
            gen_imgs.append(gen_out.cpu().data.numpy())
            
        real_imgs = np.concatenate(real_imgs, axis=0)
        gen_imgs = np.concatenate(gen_imgs, axis=0)

    return get_fid(real_imgs, gen_imgs)


def main(args):
    train_loader, test_loader = load_data(args)

    model = Model(args.model_type)
    if args.cuda:
        model = model.cuda()

    print("[model param]: {}".format(get_n_params(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)

    if not args.eval:
        global_step = 1
        if args.checkpoints is not None:
            checkpoints = torch.load(args.checkpoints)
            model.load_state_dict(checkpoints["state_dict"])
            optimizer.load_state_dict(checkpoints["optim_state_dict"])
            global_step = checkpoints["step"]
            print('Model Load: {}'.format(args.checkpoints))
        losses = 0.
        pbar = tqdm(range(global_step, args.steps + 1))
        while global_step < args.steps:
            for data, label in train_loader:
                if args.cuda:
                    data, label = data.cuda(), label.cuda()

                optimizer.zero_grad()
                loss = model._gen_loss(data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                losses += loss
                optimizer.step()
                sched.step()

                global_step += 1
                pbar.set_description("Loss: {}".format(losses / global_step))
                pbar.update(1)
                if global_step % 10000 == 0 or global_step == 2:
                    save_ckpt(global_step, model, optimizer)
                    print("[Step: {0:7d}], loss: {1:.4f}".format(global_step, losses / global_step))
    else:
        load_filename = os.path.join('checkpoints2', 'global_step_{}.pt'.format(args.step))
        checkpoints = torch.load(load_filename)

        model.load_state_dict(checkpoints["state_dict"])

        save_dir = 'save_imgs'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        x_sample = torch.randn([args.num_samples, 3, 32, 32])
        if args.cuda:
            x_sample = x_sample.cuda()
        s_num = 400
        model.eval()
        for num in range(0, args.num_samples // s_num):
            for t in tqdm(range(999, -1, -1)):
                if args.model_type == 'ddpm':
                    x_sample[num * s_num:(num + 1) * s_num] = model._infer(x_sample[num * s_num:(num + 1) * s_num], t).clamp(-1., 1.)
                else:
                    x_sample[num * s_num:(num + 1) * s_num] = model._infer(x_sample[num * s_num:(num + 1) * s_num], t)
                
                if t == 0:
                    for ii in range(num * s_num, (num + 1) * s_num):
                        result_png = 'model_{}_t_{}_num_{}.png'.format(args.step, t + 1, ii)
                        sample_img = x_sample[ii].permute(1, 2, 0).clamp(-1., 1.)
                        sample_img = ((sample_img + 1.) * 127.5).type(torch.uint8).cpu().data.numpy()
                        imageio.imwrite(os.path.join(save_dir, result_png), sample_img)
                        
        gen_data = ((x_sample.clamp(-1., 1.) + 1.) * 127.5).type(torch.uint8)
        fid = _eval(gen_data, test_loader, args)
        print('FID: ', fid)
        

if __name__ == '__main__':
    args = load_args()
    main(args)

