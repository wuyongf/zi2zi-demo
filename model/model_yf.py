import imageio.v2 as imageio
import torch
import torch.nn as nn
from .generators_yf import UNetGenerator
from .discriminators import Discriminator
from .losses import CategoryLoss, BinaryLoss
import os
from torch.optim.lr_scheduler import StepLR
from utils.init_net import init_net
import torchvision.utils as vutils
import numpy as np


class Zi2ZiModelDemo:
    def __init__(self, input_nc=3, embedding_num=40, embedding_dim=128,
                 ngf=64, ndf=64,
                 Lconst_penalty=15, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, gpu_ids=None, save_dir='.', is_training=True,
                 image_size=256):

        if is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.L1_penalty = L1_penalty

        self.schedule = schedule

        self.save_dir = save_dir
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        self.embedding_dim = embedding_dim
        self.embedding_num = embedding_num
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.is_training = is_training
        self.image_size = image_size

    def setup(self):

        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc,
            embedding_num=self.embedding_num,
            embedding_dim=self.embedding_dim,
            ngf=self.ngf,
            use_dropout=self.use_dropout
        )
        self.netD = Discriminator(
            input_nc=2 * self.input_nc,
            embedding_num=self.embedding_num,
            ndf=self.ndf,
            image_size=self.image_size
        )

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.category_loss = CategoryLoss(self.embedding_num)
        self.real_binary_loss = BinaryLoss(True)
        self.fake_binary_loss = BinaryLoss(False)
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        if self.gpu_ids:
            self.category_loss.cuda()
            self.real_binary_loss.cuda()
            self.fake_binary_loss.cuda()
            self.l1_loss.cuda()
            self.mse.cuda()
            self.sigmoid.cuda()

        if self.is_training:
            self.netD.train()
            self.netG.train()
        else:
            self.netD.eval()
            self.netG.eval()

    def set_input(self, labels, real_A, real_B):
        if self.gpu_ids:
            self.real_A = real_A.to(self.gpu_ids[0])
            self.real_B = real_B.to(self.gpu_ids[0])
            self.labels = labels.to(self.gpu_ids[0])
        else:
            self.real_A = real_A
            self.real_B = real_B
            self.labels = labels

    def forward(self, batch, basename, result_folder, label_name, cnt):

        """
        Gen-0
        """
        # generate fake_B
        self.fake_B, self.encoded_real_A, style = self.netG(self.real_A, self.labels)
        self.encoded_fake_B = self.netG(self.fake_B).view(self.fake_B.shape[0], -1)

        # """
        # Test-1 - tutorial
        # """
        #
        # # layer = 0
        # # for param_tensor in self.netG.state_dict():
        # #     layer +=1
        # #     print('layer: [', layer , '] ',param_tensor, "\t", self.netG.state_dict()[param_tensor].size())
        # # print(self.netG.state_dict())
        # # print('layer [83]:', self.netG.state_dict()['embedder.weight'])
        # # print('layer [26?]:', self.netD.state_dict()['catagory.bias'])
        #
        # # print('fake_B flag1: ', self.fake_B.shape)
        # # print('self.embedder(1):', self.netG.embedder(1))

        """
        Gen-1
        """
        # import random
        #
        # random.seed(100) #1,3
        # num = random.random()
        #
        # steps = 10
        # new_x_dim = steps + 1
        # alphas = np.linspace(0.0, 1.0, new_x_dim)
        #
        # gen1_fake_styles = []
        # gen1_fake_styles_name = []
        #
        # for i in range(0,20,1):
        #     for j in range(i+1, 20, 1):
        #
        #         # if i == 7 and j == 16 or i == 12 and j == 18:
        #
        #             count = 0
        #             for alpha in alphas:
        #                 count +=1
        #
        #                 img_name = 'gen1_s' + str(i) + "s" + str(j) + '_' + str(count)
        #
        #                 fake_style_name = 's' + str(i) + "s" + str(j) + '_' + str(count)
        #                 gen1_fake_styles_name.append(fake_style_name)
        #
        #                 style_dad = self.netG.get_style(torch.tensor([i], device='cuda:0'))
        #                 style_mon = self.netG.get_style(torch.tensor([j], device='cuda:0'))
        #
        #                 fake_style = style_dad * (1 - alpha) + style_mon * alpha
        #                 gen1_fake_styles.append(fake_style)
        #
        #                 self.fake_B, self.encoded_real_A, style = self.netG(self.real_A, fake_style, self.labels)
        #                 self.encoded_fake_B = self.netG(self.fake_B, fake_style).view(self.fake_B.shape[0], -1)
        #
        #                 # save!!!
        #                 tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
        #                 for label, image_tensor in zip(batch[0], tensor_to_plot):
        #                     result_dir = os.path.join(basename, result_folder)
        #                     result_dir2 = "gen1/" + 's' + str(i) + "s" + str(j) + '/'
        #                     result_dir3 = os.path.join(result_dir, result_dir2)
        #                     if not os.path.exists(result_dir3):
        #                         os.makedirs(result_dir3)
        #                     # chk_mkdir(label_dir)
        #                     # vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
        #                     vutils.save_image(image_tensor[:, :, 1:256], os.path.join(result_dir3, img_name + '.png'))
        #                     cnt += 1



        """
        Gen2
        """

        # img_name = 'gen2_' + gen1_fake_styles_name[0] + gen1_fake_styles_name[1]
        #
        # style_dad = gen1_fake_styles[0]
        # style_mon = gen1_fake_styles[1]
        #
        # fake_style = style_dad * num + style_mon * (1 - num)
        #
        # self.fake_B, self.encoded_real_A, style = self.netG(self.real_A, fake_style,
        #                                                     torch.tensor([0], device='cuda:0'))
        #
        # self.encoded_fake_B = self.netG(self.fake_B, fake_style, ).view(self.fake_B.shape[0], -1)
        #
        # # save!!!
        # tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
        # for label, image_tensor in zip(batch[0], tensor_to_plot):
        #     result_dir = os.path.join(basename, result_folder)
        #     result_dir2 = "gen2"
        #     chk_mkdir(result_dir2)
        #     result_dir3 = os.path.join(result_dir, result_dir2)
        #     # chk_mkdir(label_dir)
        #     # vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
        #     vutils.save_image(image_tensor[:, :, 1:256], os.path.join(result_dir3, img_name + '.png'))
        #     cnt += 1

    """
    Test - New forward function
    """
    def new_forward(self, batch, basename, result_folder, label_name, cnt, gen_no):

        # Random Seed
        import random
        random.seed(100) #1,3
        num = random.random()

        if gen_no == 0:
            #generate fake_B
            fake_style = []
            self.fake_B, self.encoded_real_A, style = self.netG(self.real_A, fake_style, gen_no, self.labels)
            self.encoded_fake_B = self.netG(self.fake_B, fake_style, gen_no).view(self.fake_B.shape[0], -1)

        if gen_no == 1:
            # 1. Interpolation
            # 2. Save Gif
            steps = 40
            new_x_dim = steps + 1
            alphas = np.linspace(0.0, 1.0, new_x_dim)

            for i in range(0,20,1):
                for j in range(i+1, 20, 1):
                    count = 0
                    images = []
                    for alpha in alphas:
                        count +=1
                        img_name = 'gen1_s' + str(i) + "s" + str(j) + '_' + str(count)
                        style_dad = self.netG.get_style(torch.tensor([i], device='cuda:0'))
                        style_mon = self.netG.get_style(torch.tensor([j], device='cuda:0'))
                        fake_style = style_dad * (1 - alpha) + style_mon * alpha
                        self.fake_B, self.encoded_real_A, style = self.netG(self.real_A, fake_style, gen_no, self.labels)
                        self.encoded_fake_B = self.netG(self.fake_B, fake_style, gen_no).view(self.fake_B.shape[0], -1)
                        # save img and gif!!!
                        tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
                        for label, image_tensor in zip(batch[0], tensor_to_plot):
                            result_dir = os.path.join(basename, result_folder)
                            result_dir2 = "gen1/" + 's' + str(i) + "s" + str(j) + '/'
                            result_dir3 = os.path.join(result_dir, result_dir2)
                            if not os.path.exists(result_dir3):
                                os.makedirs(result_dir3)
                            image_name = os.path.join(result_dir3, img_name + '.png')
                            vutils.save_image(image_tensor[:, :, 1:256], image_name)
                            images.append(imageio.imread(image_name))
                            cnt += 1
                    # save as gif!!!
                    result_img_addr =  os.path.join(result_dir3, 'gen1_s' + str(i) + "s" + str(j) + '_transform.gif')
                    imageio.mimsave(result_img_addr, images, fps=10)

        if gen_no == 2:

            gen1_fake_styles = []
            gen1_fake_styles_name = []

            for i in range(0,20,1):
                for j in range(i+1, 20, 1):
                        count = 0
                        for alpha in alphas:
                            count +=1

                            img_name = 'gen1_s' + str(i) + "s" + str(j) + '_' + str(count)

                            fake_style_name = 's' + str(i) + "s" + str(j) + '_' + str(count)
                            gen1_fake_styles_name.append(fake_style_name)

                            style_dad = self.netG.get_style(torch.tensor([i], device='cuda:0'))
                            style_mon = self.netG.get_style(torch.tensor([j], device='cuda:0'))

                            fake_style = style_dad * (1 - alpha) + style_mon * alpha
                            gen1_fake_styles.append(fake_style)

            img_name = 'gen2_' + gen1_fake_styles_name[0] + gen1_fake_styles_name[1]

            style_dad = gen1_fake_styles[0]
            style_mon = gen1_fake_styles[1]

            fake_style = style_dad * num + style_mon * (1 - num)

            self.fake_B, self.encoded_real_A, style = self.netG(self.real_A, fake_style, gen_no, torch.tensor([0], device='cuda:0'))
            self.encoded_fake_B = self.netG(self.fake_B, fake_style, ).view(self.fake_B.shape[0], -1)

            # save!!!
            tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
            for label, image_tensor in zip(batch[0], tensor_to_plot):
                result_dir = os.path.join(basename, result_folder)
                result_dir2 = "gen2"
                chk_mkdir(result_dir2)
                result_dir3 = os.path.join(result_dir, result_dir2)
                # chk_mkdir(label_dir)
                # vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
                vutils.save_image(image_tensor[:, :, 1:256], os.path.join(result_dir3, img_name + '.png'))
                cnt += 1e6

    def backward_D(self, no_target_source=False):

        real_AB = torch.cat([self.real_A, self.real_B], 1)
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)

        real_D_logits, real_category_logits = self.netD(real_AB)
        fake_D_logits, fake_category_logits = self.netD(fake_AB.detach())

        real_category_loss = self.category_loss(real_category_logits, self.labels)
        fake_category_loss = self.category_loss(fake_category_logits, self.labels)
        category_loss = (real_category_loss + fake_category_loss) * self.Lcategory_penalty

        d_loss_real = self.real_binary_loss(real_D_logits)
        d_loss_fake = self.fake_binary_loss(fake_D_logits)

        self.d_loss = d_loss_real + d_loss_fake + category_loss / 2.0
        self.d_loss.backward()
        return category_loss

    def backward_G(self, no_target_source=False):

        fake_AB = torch.cat([self.real_A, self.fake_B], 1)
        fake_D_logits, fake_category_logits = self.netD(fake_AB)

        # encoding constant loss
        # this loss assume that generated imaged and real image should reside in the same space and close to each other
        const_loss = self.Lconst_penalty * self.mse(self.encoded_real_A, self.encoded_fake_B)
        # L1 loss between real and generated infer_images
        l1_loss = self.L1_penalty * self.l1_loss(self.fake_B, self.real_B)
        fake_category_loss = self.Lcategory_penalty * self.category_loss(fake_category_logits, self.labels)

        cheat_loss = self.real_binary_loss(fake_D_logits)

        self.g_loss = cheat_loss + l1_loss + fake_category_loss + const_loss
        self.g_loss.backward()
        return const_loss, l1_loss, cheat_loss

    def update_lr(self):
        # There should be only one param_group.
        for p in self.optimizer_D.param_groups:
            current_lr = p['lr']
            update_lr = current_lr / 2.0
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.0002)
            p['lr'] = update_lr
            print("Decay net_D learning rate from %.5f to %.5f." % (current_lr, update_lr))

        for p in self.optimizer_G.param_groups:
            current_lr = p['lr']
            update_lr = current_lr / 2.0
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.0002)
            p['lr'] = update_lr
            print("Decay net_G learning rate from %.5f to %.5f." % (current_lr, update_lr))

    def optimize_parameters(self):
        self.forward()  # compute fake infer_images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        category_loss = self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights

        # magic move to Optimize G again
        # according to https://github.com/carpedm20/DCGAN-tensorflow
        # collect all the losses along the way
        self.forward()  # compute fake infer_images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        const_loss, l1_loss, cheat_loss = self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights
        return const_loss, l1_loss, category_loss, cheat_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in ['G', 'D']:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    # torch.save(net.cpu().state_dict(), save_path)
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    net.load_state_dict(torch.load(load_path))
                else:
                    net.load_state_dict(torch.load(load_path,map_location=torch.device('cpu')))
                # net.eval()
        print('load model %d' % epoch)

    def sample(self, batch, basename):
        chk_mkdir(basename)
        cnt = 0
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
            for label, image_tensor in zip(batch[0], tensor_to_plot):
                label_dir = os.path.join(basename, str(label.item()))
                chk_mkdir(label_dir)
                vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
                cnt += 1
            # img = vutils.make_grid(tensor_to_plot)
            # vutils.save_image(tensor_to_plot, basename + "_construct.png")
            '''
            We don't need generate_img currently.
            self.set_input(torch.randn(1, self.embedding_num).repeat(batch[0].shape[0], 1), batch[2], batch[1])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_B, self.real_A], 3)
            vutils.save_image(tensor_to_plot, basename + "_generate.png")
            '''

    def sample_demo(self, batch, basename):
        chk_mkdir(basename)
        cnt = 0
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
            for label, image_tensor in zip(batch[0], tensor_to_plot):
                label_dir = os.path.join(basename, str(label.item()))
                chk_mkdir(label_dir)
                vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
                cnt += 1
            # img = vutils.make_grid(tensor_to_plot)
            # vutils.save_image(tensor_to_plot, basename + "_construct.png")
            '''
            We don't need generate_img currently.
            self.set_input(torch.randn(1, self.embedding_num).repeat(batch[0].shape[0], 1), batch[2], batch[1])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_B, self.real_A], 3)
            vutils.save_image(tensor_to_plot, basename + "_generate.png")
            '''
    def sample_demo_yf(self, batch, basename, result_folder, label_name, gen_no):
        
        chk_mkdir(basename)
        chk_mkdir(os.path.join(basename,result_folder))
        cnt = 0
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1])
            self.new_forward(batch, basename, result_folder, label_name, cnt, gen_no)

            """
            Comment for Test-1 / Gen-1 & Gen-2
            """
            if gen_no == 0:
                tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
                for label, image_tensor in zip(batch[0], tensor_to_plot):
                    result_dir = os.path.join(basename, result_folder)
                    # chk_mkdir(label_dir)
                    # vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
                    vutils.save_image(image_tensor, os.path.join(result_dir, label_name + '.png'))
                    cnt += 1
                img = vutils.make_grid(tensor_to_plot)
                print(f'shape: {tensor_to_plot.shape}')
                vutils.save_image(tensor_to_plot, basename + "_construct.png")

            '''
            We don't need generate_img currently.
            self.set_input(torch.randn(1, self.embedding_num).repeat(batch[0].shape[0], 1), batch[2], batch[1])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_B, self.real_A], 3)
            vutils.save_image(tensor_to_plot, basename + "_generate.png")
            '''
            # return style
    
    def web_demo_sample(self, batch, basename, result_folder, label_name, gen_no):
        from PIL import Image
        def split_image_to_half(img_path):
            # Read the image
            image = Image.open(img_path)
            # Get the size of the image
            width, height = image.size
            # Calculate the dimensions for each half
            left_half = (0, 0, width // 2, height)
            right_half = (width // 2, 0, width, height)
            # Crop the image to get the left half
            left_image = image.crop(left_half)
            # Crop the image to get the right half
            right_image = image.crop(right_half)
            return left_image, right_image
        
        chk_mkdir(basename)
        cnt = 0
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1])
            self.new_forward(batch, basename, result_folder, label_name, cnt, gen_no)

            """
            Comment for Test-1 / Gen-1 & Gen-2
            """
            if gen_no == 0:
                tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
                for label, image_tensor in zip(batch[0], tensor_to_plot):
                    result_dir = os.path.join(basename, result_folder)
                    # chk_mkdir(label_dir)
                    # vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
                    vutils.save_image(image_tensor, os.path.join(result_dir, label_name + '.png'))
                    cnt += 1
                img = vutils.make_grid(tensor_to_plot)
                vutils.save_image(tensor_to_plot, basename + "_construct.png")

                infer_img , real_image = split_image_to_half(basename + "_construct.png")
                
                return infer_img, real_image

            '''
            We don't need generate_img currently.
            self.set_input(torch.randn(1, self.embedding_num).repeat(batch[0].shape[0], 1), batch[2], batch[1])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_B, self.real_A], 3)
            vutils.save_image(tensor_to_plot, basename + "_generate.png")
            '''
            # return style

def chk_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)