import os
import time
import datetime
import torch
import torch.nn as nn
import os.path as osp

from model import Generator
from model import Discriminator
from torchvision.utils import save_image


class Solver(object):

    def __init__(self, config, data_loader):
        """Initialize configurations."""
        self.data_loader = data_loader
        self.img_size    = config['MODEL_CONFIG']['IMG_SIZE']
        assert self.img_size in [256]

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.g_lr          = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr          = float(config['TRAINING_CONFIG']['D_LR'])
        self.lambda_cls = config['TRAINING_CONFIG']['LAMBDA_CLS']
        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_g_recon = config['TRAINING_CONFIG']['LAMBDA_G_RECON']
        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_d_gp     = config['TRAINING_CONFIG']['LAMBDA_GP']
        self.d_critic      = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic      = config['TRAINING_CONFIG']['G_CRITIC']
        self.mse_loss = nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.cls_loss = nn.BCEWithLogitsLoss()

        self.gan_loss = config['TRAINING_CONFIG']['GAN_LOSS']
        self.num_cls = config['TRAINING_CONFIG']['NUM_CLS']
        assert self.gan_loss in ['lsgan', 'wgan', 'vanilla']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']
        if self.gan_loss == 'lsgan':
            self.adversarial_loss = torch.nn.MSELoss()
        elif self.gan_loss =='vanilla':
            self.adversarial_loss = torch.nn.BCELoss()

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.g_spec = config['TRAINING_CONFIG']['G_SPEC'] == 'True'
        self.d_spec = config['TRAINING_CONFIG']['D_SPEC'] == 'True'

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD'] == 'True'

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']

        self.build_model()

        if self.use_tensorboard == 'True':
            self.build_tensorboard()

    def build_model(self):

        self.G = Generator(spec_norm=self.g_spec, LR=0.02, num_attr=self.num_cls).to(self.gpu)
        self.D = Discriminator(spec_norm=self.d_spec, LR=0.02, num_attr=self.num_cls).to(self.gpu)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr, c_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.gpu)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def get_state_dict(self, path):

        if path.startswith("module-"):
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(model_path, map_location=self.device)
            # https://github.com/computationalmedia/semstyle/issues/3
            # https://github.com/pytorch/pytorch/issues/10622
            # https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666/2
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            return new_state_dict
        else:
            return torch.load(path, map_location=lambda storage, loc: storage)

    def restore_model(self, epoch):

        G_path = osp.join(self.model_dir, '*{}-G.ckpt'.format(epoch))
        D_path = osp.join(self.model_dir, '*{}-D.ckpt'.format(epoch))

        self.G.load_state_dict(self.get_state_dict(G_path))
        self.D.load_state_dict(self.get_state_dict(D_path))

    def train(self):

        # Set data loader.
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        fixed_image, fixed_attr_s = next(data_iter)

        fixed_image = fixed_image.to(self.gpu)
        fixed_attr_s = fixed_attr_s.to(self.gpu)

        rand_idx = torch.randperm(fixed_attr_s.size(0))
        fixed_attr_t1 = fixed_attr_s[rand_idx]
        fixed_attr_t1 = fixed_attr_t1.to(self.gpu)

        rand_idx = torch.randperm(fixed_attr_s.size(0))
        fixed_attr_t2 = fixed_attr_s[rand_idx]
        fixed_attr_t2 = fixed_attr_t2.to(self.gpu)

        rand_idx = torch.randperm(fixed_attr_s.size(0))
        fixed_attr_t3 = fixed_attr_s[rand_idx]
        fixed_attr_t3 = fixed_attr_t3.to(self.gpu)

        rand_idx = torch.randperm(fixed_attr_s.size(0))
        fixed_attr_t4 = fixed_attr_s[rand_idx]
        fixed_attr_t4 = fixed_attr_t4.to(self.gpu)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_time = time.time()
        print('Start training...')
        for e in range(self.epoch):

            for i in range(iterations):
                try:
                    image, attr_s = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    image, attr_s = next(data_iter)

                image = image.to(self.gpu)
                attr_s = attr_s.to(self.gpu)

                rand_idx = torch.randperm(attr_s.size(0))
                attr_t = attr_s[rand_idx]

                loss_dict = dict()
                if (i + 1) % self.d_critic == 0:

                    fake_images = self.G(image, attr_t - attr_s)

                    real_score, pred = self.D(image)
                    fake_score, _ = self.D(fake_images.detach())

                    d_cls_loss = self.lambda_cls * self.cls_loss(pred, attr_s)

                    if self.gan_loss in ['lsgan', 'vanilla']:
                        d_loss_real = self.adversarial_loss(real_score, torch.ones_like(real_score))
                        d_loss_fake = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
                        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake + d_cls_loss
                    elif self.gan_loss == 'wgan':
                        d_loss_real = -torch.mean(real_score)
                        d_loss_fake = torch.mean(fake_score)
                        alpha = torch.rand(fake_images.size(0), 1, 1, 1).to(self.gpu)
                        x_hat = (alpha * fake_images.data + (1 - alpha) * fake_images.data).requires_grad_(True)
                        out_src, _ = self.D(x_hat)
                        d_loss_gp = self.gradient_penalty(out_src, x_hat)
                        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake + self.lambda_d_gp * d_loss_gp + d_cls_loss

                    if torch.isnan(d_loss) or torch.isinf(d_loss):
                        print('d_loss is nan or inf')
                        return

                    # Backward and optimize.
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Logging.
                    loss_dict['D/loss_real'] = self.lambda_d_real * d_loss_real.item()
                    loss_dict['D/loss_fake'] = self.lambda_d_fake * d_loss_fake.item()
                    loss_dict['D/loss_cls'] = d_cls_loss.item()

                    if self.gan_loss == 'wgan':
                        loss_dict['D/local_loss_gp'] = self.lambda_d_gp * d_loss_gp.item()

                if (i + 1) % self.g_critic == 0:

                    fake_images = self.G(image, attr_t - attr_s)
                    fake_score, pred = self.D(fake_images)

                    g_cls_loss = self.cls_loss(pred, attr_t)

                    if self.gan_loss in ['lsgan', 'vanilla']:
                        g_loss_fake = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
                    elif self.gan_loss == 'wgan':
                        g_loss_fake = - torch.mean(fake_score)
                    else:
                        pass

                    g_loss_recon = self.l1_loss(self.G(image, torch.zeros_like(attr_t)), image)

                    g_loss = self.lambda_g_fake * g_loss_fake + \
                             self.lambda_g_recon * g_loss_recon + \
                             self.lambda_cls * g_cls_loss

                    if torch.isnan(g_loss) or torch.isinf(g_loss):
                        print('g_loss is nan or inf')
                        return

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss_dict['G/loss_fake'] = self.lambda_g_fake * g_loss_fake.item()
                    loss_dict['G/loss_recon'] = self.lambda_g_recon * g_loss_recon.item()
                    loss_dict['G/loss_cls'] = self.lambda_cls * g_cls_loss.item()

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    image_report = list()
                    image_report.append(fixed_image)
                    image_report.append(self.G(fixed_image, torch.zeros_like(fixed_attr_s)))
                    image_report.append(self.G(fixed_image, fixed_attr_t1 - fixed_attr_s))
                    image_report.append(self.G(fixed_image, fixed_attr_t2 - fixed_attr_s))
                    image_report.append(self.G(fixed_image, fixed_attr_t3 - fixed_attr_s))
                    image_report.append(self.G(fixed_image, fixed_attr_t4 - fixed_attr_s))
                    x_concat = torch.cat(image_report, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(e + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(self.sample_dir))
            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(e + 1))
                D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(e + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

        print('Training is finished')

    def test(self):
        pass

