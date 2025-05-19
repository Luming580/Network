from datetime import datetime
from Gabor.Net import Generator, Discriminator
import random
from Run.Vgg_loss import *
from datasets import get_loader, test_dataset
from Run.option import opt
import logging
from Run.src import adjust_lr, limit_zero2one
from Run.Loggers import Logger
import piq
import os
from torchvision.utils import save_image
from tqdm import tqdm
from nets.Net import classifiernet
from torch.cuda.amp import GradScaler as GradScaler
from nets.net_training import (NetTrainer, get_lr_scheduler,
                               set_optimizer_lr)
from utils.utils import (get_classes, show_config)
from Run.autotest import calculate_average_ssim

save_model_path = r"~/"
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

file_handler = logging.FileHandler(save_model_path + 'log_dg_net.log', mode='a', encoding='utf-8')
formatter = logging.Formatter('[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                              datefmt='%Y-%m-%d %I:%M:%S %p')
file_handler.setFormatter(formatter)
logging.getLogger('').addHandler(file_handler)
logging.getLogger('').setLevel(logging.INFO)
logging.info("Train")
logging.info("Config")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

Generator = Generator()
Discriminator = Discriminator()
perceptual_loss = Perceptual_loss134()
criterion_GAN = torch.nn.L1Loss()
criterion_L1 = torch.nn.L1Loss()
criterion_L2 = torch.nn.MSELoss()

if cuda:
    Generator = Generator.cuda()
    Discriminator = Discriminator.cuda()
    criterion_GAN.cuda()
    criterion_L1.cuda()
    criterion_L2.cuda()

generator_losses = []
discriminator_losses = []

optimizer_G = torch.optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002)
optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=0.00002, betas=(0.5, 0.999), weight_decay=0.0002)
logging.info('Generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002')
logging.info('Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002')

# set path
image_root = opt.original_image_root
target_image_root = opt.target_image_root

patch = (1, opt.trainsize // 2 ** 4, opt.trainsize // 2 ** 4)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

val_image_root = opt.val_original_image_root
val_target_image_root = opt.val_target_image_root
val_gt_root = opt.val_gt_root
val_loader = test_dataset(val_image_root, val_target_image_root, val_gt_root, opt.testsize)

best_ssim = 0
best_ssim_epoch = 0

###########################################
###   Detect Network
##########################################
classes_path = 'model_data/voc_classes.txt'
model_path = 'model_data/voc_weights_resnet.pth'
input_shape = [256, 256]
backbone = "resnet50"
pretrained = False
anchors_size = [8, 16, 32]
Init_lr = 1e-4
Min_lr = Init_lr * 0.01
optimizer_type = "adam"
momentum = 0.5
weight_decay = 0
lr_decay_type = 'cos'

train_annotation_path = '2007_train.txt'

class_names, num_classes = get_classes(classes_path)
train_gpu = [0, ]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
ngpus_per_node = len(train_gpu)
print('Number of devices: {}'.format(ngpus_per_node))
Detect = classifiernet(num_classes, anchor_scales=anchors_size, backbone=backbone, pretrained=pretrained)

scaler = GradScaler()
Detect_train = Detect.train()
if True:
    Detect_train = torch.nn.DataParallel(Detect_train)
    torch.backends.cudnn.benchmark = True
    Detect_train = Detect_train.cuda()
with open(train_annotation_path, encoding='utf-8') as f:
    train_lines = f.readlines()

num_train = len(train_lines)

show_config(
    classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
    Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type, \
    num_train=num_train)

nbs = 16
lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
Init_lr_fit = min(max(opt.batchsize / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(opt.batchsize / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

optimizer = {
    'adam': torch.optim.Adam(Detect.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
    'sgd': torch.optim.SGD(Detect.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                           weight_decay=weight_decay)
}[optimizer_type]
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, opt.n_epochs)


dataloader = get_loader(image_root, target_image_root, train_lines, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(dataloader)
logger = Logger(opt.n_epochs, len(dataloader))
train_util = NetTrainer(Detect_train, optimizer)


def train(dataloader, Generator, Discriminator, epoch, save_model_path, optimizer_G, optimizer_D):
    global best_train_epoch

    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    Generator.train()
    G_loss = 0
    D_loss = 0
    for i, (original_images, target_images, boxes, labels) in enumerate(dataloader, start=1):
        optimizer_G.zero_grad()
        # --------------------------------------------------------
        # Model inputs
        # --------------------------------------------------------
        target_images = target_images.cuda()
        original_images = original_images.cuda()

        # Adversarial ground truths
        valid = torch.ones((target_images.size(0), *patch), dtype=torch.float32, device=target_images.device,
                           requires_grad=False)
        fake = torch.zeros((target_images.size(0), *patch), dtype=torch.float32, device=target_images.device,
                           requires_grad=False)

        # --------------------------------------------------------
        #  Train Generators
        # --------------------------------------------------------

        mid_fake_target = Generator(original_images)
        # 生成器gan损失
        pred_fake = Discriminator(original_images, mid_fake_target)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # L1损失
        loss_mae = criterion_L1(mid_fake_target, target_images)
        loss_mse = criterion_L2(mid_fake_target, target_images)

        # vgg loss
        vgg_loss = perceptual_loss(mid_fake_target, target_images)

        # --------------------------------------------------------
        #  Train Detect
        # --------------------------------------------------------

        rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(mid_fake_target, boxes, labels, 1, fp16=False,
                                                                          scaler=scaler)

        g_loss = 2 * loss_GAN + (loss_mse + loss_mae) * 150 \
                 + vgg_loss * 100 \
                 + 0.02 * roi_loc + 0.02 * roi_cls
        g_loss.backward()

        # clip_gradient(optimizer_G, opt.clip)
        optimizer_G.step()
        G_loss += g_loss.item()

        # --------------------------------------------------------
        #  Train Discriminator
        # --------------------------------------------------------
        optimizer_D.zero_grad()
        # Real loss
        pred_fake = Discriminator(original_images, target_images)
        loss_D_fake = criterion_GAN(pred_fake, fake)
        # Fake loss
        pred_real = Discriminator(original_images, mid_fake_target.detach())
        loss_D_real = criterion_GAN(pred_real, valid)

        # Total discriminator loss
        disc_loss = (loss_D_fake + loss_D_real) * 0.5
        disc_loss.backward()
        optimizer_D.step()
        D_loss += disc_loss.item()

        # --------------------------------------------------------
        #  Logging Progress
        # --------------------------------------------------------
        limit_fake_target = limit_zero2one(mid_fake_target)
        limit_real_target = limit_zero2one(target_images)
        ssim = piq.ssim(limit_fake_target, limit_real_target, data_range=1.).cpu()
        # ---------------------------------------------------------------------------------
        logger.log({
            'loss_GAN': loss_GAN,
            'loss_mae': loss_mae,
            'loss_mse': loss_mse,
            'vgg_loss': vgg_loss,
            'g_loss': g_loss,
            'ssim': ssim,
        }, )
        logging.info(
            '#TRAIN#{}:Epoch [{:03d}/{:03d}], Step [{:04d}], gan损失为: {:.4f}, loss_mae为: {:.4f}, loss_mse为: {:.4f}, vgg_loss为: {:.4f}, g_loss:{:.4f}'.
            format(datetime.now(), epoch, opt.n_epochs, i, loss_GAN, loss_mae, loss_mse, vgg_loss, g_loss))

    logging.info("\n=============================================")
    logging.info('Epoch %d 生成器loss: %.10f' % (epoch, G_loss / (i + 1)))
    logging.info('Epoch %d 鉴别器loss: %.10f' % (epoch, D_loss / (i + 1)))
    logging.info("=============================================\n")

    # --------------------------------------------------------
    #  Save model
    # --------------------------------------------------------

    if epoch > 90 or (epoch > 50 and epoch % 5 == 0):
        torch.save(Generator.state_dict(),
                   f"{save_model_path}/Generator_%d.pth" % (epoch))

best_ssim = 0

def valiable(val_loader, Generator, epoch, save_model_path):
    global best_ssim, best_ssim_epoch
    save_img_pth = r"~/"
    Generator = Generator()

    save_path_generator = save_img_pth + "fakeimg2"

    if not os.path.exists(save_path_generator):
        os.makedirs(save_path_generator)

    with torch.no_grad():
        for i in tqdm(range(val_loader.size), desc="processing image...", unit="file"):
            original_images, target_images, tar, gt, name, g = val_loader.load_data()
            name = name.replace(".png", "")
            original_images = original_images.cuda()
            res = Generator(original_images)
            img_sample1 = res.data
            save_image(img_sample1, f"{save_path_generator}/{name}.png", nrow=5, normalize=True)

    aver_ssims, std_ssim = calculate_average_ssim(save_path_generator, val_target_image_root)
    if aver_ssims > best_ssim:
        best_ssim = aver_ssims
        best_ssim_epoch = epoch
        torch.save(Generator.state_dict(), f"{save_model_path}/Generator_%d.pth" % (epoch))


if __name__ == '__main__':
    print("Start train...")
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed(17)
    for epoch in range(1, opt.n_epochs + 1):
        cur_lr = adjust_lr(optimizer_G, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(dataloader, Generator, Discriminator, epoch, save_model_path, optimizer_G, optimizer_D)
        if epoch % 10 == 0:
            valiable(val_loader, Generator, epoch, save_model_path)
