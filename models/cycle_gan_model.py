import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

##################################### CODE ADDED / START #####################################
from torchsummary import summary

import numpy as np
from random import random
import torch.nn.functional as F

# 점점 비율을 키워서 사용
def warmup(start, end, max_steps, current_step):
    if current_step > max_steps:
        return end
    return (end - start) * (current_step / max_steps) + start

# Cross-Consistency-Loss 계산시에 사용    
def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target

# cutmix 샘플 생성시 사용 : real과 fake를 이용하여 생성함
def cutmix(source, target, coors, alpha = 1.):
    source, target = map(torch.clone, (source, target))
    ((y0, y1), (x0, x1)), _ = coors
    source[:, :, y0:y1, x0:x1] = target[:, :, y0:y1, x0:x1]
    return source

def cutmix_coordinates(height, width, alpha = 1.):
    lam = np.random.beta(alpha, alpha)

    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))

    return ((y0, y1), (x0, x1)), lam

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException
        
class NanException(Exception):
    pass
###################################### CODE ADDED / END ######################################

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        
        Generators: 
        1) G_A : A -> B
        2) G_B : B -> A
        
        Discriminators: 
        1) D_A : G_A(A) vs. B
        2) D_B : G_B(B) vs. A

        Forward cycle loss:
        : lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        
        Backward cycle loss:
        : lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)

        Identity loss (optional)
        : lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        self.netD_losses = []

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # 1) 입력채널       / 2) 출력채널 / 3) 마지막 필터 개수 / 4) 구조 : resnet_9blocks 
        # 5) instance norm / 6) dropout / 7) normal init     / 8) scaling factor
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            # 입력채널수 / 첫번째 필터 개수 / 구조 : basic / 레이어개수 / 
            # instance norm / normal init / scaling factor / gpu_id
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc) # 입출력 channel 갯수 동일하게 설정
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions

            # opt.gan_mode : lsgan by default
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake, opt, total_iters):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        if opt.netD != 'unet':
            # 기존의 코드 주석처리
            pred_real = netD(real)
            loss_D_real = self.criterionGAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterionGAN(pred_fake, False)
            # Combined loss and calculate gradients
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            self.netD_losses = [pred_real.shape, pred_fake.shape]
        elif opt.netD == 'unet':
        ##################################### CODE ADDED / START #####################################

            dec_loss_coef = warmup(0, 1., 30000, total_iters)
            cutmix_prob   = warmup(0, 0.25, 30000, total_iters)
            apply_cutmix  = random() < cutmix_prob
            cr_weight     = 0.2


            
            (real_enc_out, real_dec_out) = netD(real)
            (fake_enc_out, fake_dec_out) = netD(fake.detach())

            loss_D_real_enc = self.criterionGAN(real_enc_out, True)
            loss_D_real_dec = self.criterionGAN(real_dec_out, True)

            loss_D_fake_enc = self.criterionGAN(fake_enc_out, False)
            loss_D_fake_dec = self.criterionGAN(fake_dec_out, False)
            
            disc_loss = (loss_D_real_enc + loss_D_fake_enc) + (loss_D_real_dec + loss_D_fake_dec) * dec_loss_coef
            

            # loss_D_real = self.criterionGAN(real_enc_out, True)


            self.netD_losses = []

            # enc_divergence = (F.relu(1 + real_enc_out) + F.relu(1 - fake_enc_out)).mean()
            # dec_divergence = (F.relu(1 + real_dec_out) + F.relu(1 - fake_dec_out)).mean()
            # disc_loss = enc_divergence + dec_divergence * dec_loss_coef
            


            if apply_cutmix:

                mask = cutmix(
                    torch.ones_like(real_dec_out),
                    torch.zeros_like(real_dec_out),
                    cutmix_coordinates(opt.crop_size, opt.crop_size)
                            )

                if random() > 0.5:
                    mask = 1 - mask

                cutmix_images = mask_src_tgt(real, fake, mask)
                cutmix_enc_out, cutmix_dec_out = netD(cutmix_images.detach())

                # cutmix_enc_divergence = F.relu(1 - cutmix_enc_out).mean()
                # cutmix_dec_divergence =  F.relu(1 + (mask * 2 - 1) * cutmix_dec_out).mean()

                loss_D_cutmix_enc = self.criterionGAN(cutmix_enc_out, False)
                loss_D_cutmix_dec = self.criterionGAN(cutmix_dec_out, False)

                disc_loss = disc_loss + (loss_D_cutmix_enc + loss_D_cutmix_dec * dec_loss_coef)

                cr_cutmix_dec_out = mask_src_tgt(real_dec_out, fake_dec_out, mask)
                cr_loss = F.mse_loss(cutmix_dec_out, cr_cutmix_dec_out) * cr_weight

                disc_loss = disc_loss + cr_loss * dec_loss_coef

                
                # self.netD_losses = [enc_divergence, dec_divergence, cutmix_enc_out, cutmix_dec_out, cutmix_enc_divergence, cutmix_dec_divergence, cr_loss]

                disc_loss.register_hook(raise_if_nan)
            loss_D = disc_loss
        ###################################### CODE ADDED / END ######################################
        
        loss_D.backward()
        return loss_D

    # add_kyugorihtm : backward_D_basic 에서 opt를 사용하기 위해 추가
    def backward_D_A(self, opt, total_iters):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, opt, total_iters)

    # add_kyugorihtm : backward_D_basic 에서 opt를 사용하기 위해 추가
    def backward_D_B(self, opt, total_iters):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, opt, total_iters)

    def backward_G(self, opt):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0


        if opt.netD !='unet':

            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        elif opt.netD == 'unet':

            ##################################### CODE ADDED / START #####################################
            # GAN loss D_A(G_A(A))
            (fake_enc_output_A, fake_dec_output_A) = self.netD_A(self.fake_B)
            
            # self.loss_G_A = fake_enc_output_A.mean() + F.relu(1 + fake_dec_output_A).mean()
            # GAN loss D_B(G_B(B))
            (fake_enc_output_B, fake_dec_output_B) = self.netD_B(self.fake_A)
            # self.loss_G_B = fake_enc_output_B.mean() + F.relu(1 + fake_dec_output_B).mean()
            
            self.loss_G_A = self.criterionGAN(fake_enc_output_A, True) + self.criterionGAN(fake_dec_output_A, True)
            self.loss_G_B = self.criterionGAN(fake_enc_output_B, True) + self.criterionGAN(fake_dec_output_B, True)

            ###################################### CODE ADDED / END ######################################


        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B        
        self.loss_G.register_hook(raise_if_nan)
        self.loss_G.backward()


    # add_kyugorihtm : backward_D_A/backward_D_B 에서 opt를 사용하기 위해 추가
    def optimize_parameters(self, opt, total_iters):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(opt)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A(opt, total_iters)      # calculate gradients for D_A
        self.backward_D_B(opt, total_iters)      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
