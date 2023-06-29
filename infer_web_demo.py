# encoding: utf-8

from data import DatasetFromObj
from model import Zi2ZiModelDemo
from model.model import chk_mkdir
# from fgmodel.zi2zi.data import DatasetFromObj
# from fgmodel.zi2zi.model import Zi2ZiModelDemo
# from fgmodel.zi2zi.model.model import chk_mkdir
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import torch
import random
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import time
from torchsummary import summary

writer_dict = {
        '智永': 0, ' 隸書-趙之謙': 1, '張即之': 2, '張猛龍碑': 3, '柳公權': 4, '標楷體-手寫': 5, '歐陽詢-九成宮': 6,
        '歐陽詢-皇甫誕': 7, '沈尹默': 8, '美工-崩雲體': 9, '美工-瘦顏體': 10, '虞世南': 11, '行書-傅山': 12, '行書-王壯為': 13,
        '行書-王鐸': 14, '行書-米芾': 15, '行書-趙孟頫': 16, '行書-鄭板橋': 17, '行書-集字聖教序': 18, '褚遂良': 19, '趙之謙': 20,
        '趙孟頫三門記體': 21, '隸書-伊秉綬': 22, '隸書-何紹基': 23, '隸書-鄧石如': 24, '隸書-金農': 25,  '顏真卿-顏勤禮碑': 26,
        '顏真卿多寶塔體': 27, '魏碑': 28
    }

parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('--experiment_dir', required=False,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--start_from', type=int, default=0)
parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--obj_path', type=str, default='./experiment/data/val.obj', help='the obj file you infer')
parser.add_argument('--input_nc', type=int, default=1)

parser.add_argument('--from_txt', action='store_true')
parser.add_argument('--src_txt', type=str, default='大威天龍大羅法咒世尊地藏波若諸佛')
parser.add_argument('--canvas_size', type=int, default=256)
parser.add_argument('--char_size', type=int, default=256)
parser.add_argument('--run_all_label', action='store_true')
parser.add_argument('--label', type=int, default=0)
parser.add_argument('--src_font', type=str, default='charset/gbk/方正新楷体_GBK(完整).TTF')
parser.add_argument('--type_file', type=str, default='type/宋黑类字符集.txt')

parser.add_argument('--result_folder', required=False)
parser.add_argument('--infer_name', required=False)
parser.add_argument('--gen_no', type=int, required=False)


def yf_input_original_char(img_addr):
    target_img = Image.open(img_addr).convert('L')
    newsize = (256, 256)
    target_img = target_img.resize(newsize)
    return target_img

def draw_single_char(ch, font, canvas_size):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, (0, 0, 0), font=font)
    img = img.convert('L')
    return img


def latent_lerp(gan, z0, z1, nb_frames):
    """Interpolate between two images in latent space"""

    imgs = []
    for i in range(nb_frames):
        alpha = i / nb_frames
        z = (1 - alpha) * z0 + alpha * z1
        imgs.append(gan.generate_img(z))
    return imgs

def interpolate(self, source_obj, between, steps):

    # new interpolated dimension
    new_x_dim = steps + 1
    alphas = np.linspace(0.0, 1.0, new_x_dim)

    def _interpolate_tensor(_tensor):
        """
        Comupute the interpolated tensor here
        """
        x = _tensor[between[0]]
        y = _tensor[between[1]]

        interpolated = list()
        for alpha in alphas:
            interpolated.append(x * (1. - alpha) + alpha * y)

        interpolated = np.asarray(interpolated, dtype=np.float32)
        return interpolated

def main():
    args = parser.parse_args()
    data_dir = os.path.join(args.experiment_dir, "useful_functions/data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    chk_mkdir(infer_dir)

    result_folder = args.result_folder
    infer_name = args.infer_name
    gen_no = args.gen_no
    style = torch.zeros(1, 128)

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    t0 = time.time()

    model = Zi2ZiModelDemo(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    t1 = time.time()

    if args.from_txt:
        src = args.src_txt
        font = ImageFont.truetype(args.src_font, size=args.char_size)

        # author codes
        img_list = [transforms.Normalize(0.5, 0.5)(
            transforms.ToTensor()(
                draw_single_char(ch, font, args.canvas_size)
            )
        ).unsqueeze(dim=0) for ch in src]

        label_list = [args.label for _ in src]

        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor(label_list)

        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    else:
        val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'),
                                     input_nc=args.input_nc,
                                     start_from=args.start_from)
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    global_steps = 0
    with open(args.type_file, 'r', encoding='utf-8') as fp:
        fonts = [s.strip() for s in fp.readlines()]
    writer_dict = {v: k for k, v in enumerate(fonts)}

    for batch in dataloader:
        if args.run_all_label:
            # global writer_dict
            writer_dict_inv = {v: k for k, v in writer_dict.items()}
            for label_idx in range(29):
                model.set_input(torch.ones_like(batch[0]) * label_idx, batch[2], batch[1])
                model.forward()
                tensor_to_plot = torch.cat([model.fake_B, model.real_B], 3)
                # img = vutils.make_grid(tensor_to_plot)
                save_image(tensor_to_plot, os.path.join(infer_dir, "infer_{}".format(writer_dict_inv[label_idx]) + "_construct.png"))
        else:
            model.sample_demo_yf(batch, infer_dir, result_folder, infer_name, gen_no)
            global_steps += 1
            # print(batch[0]) # yf: the label --- batch[0]

    t_finish = time.time()

    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))

def infer_webdemo(args):

    # args = parser.parse_args()
    data_dir = os.path.join(args.experiment_dir, "useful_functions/data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    chk_mkdir(infer_dir)

    result_folder = args.result_folder
    infer_name = args.infer_name
    gen_no = args.gen_no
    style = torch.zeros(1, 128)

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    t0 = time.time()

    model = Zi2ZiModelDemo(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    t1 = time.time()

    if args.from_txt:
        src = args.src_txt
        font = ImageFont.truetype(args.src_font, size=args.char_size)

        # author codes
        img_list = [transforms.Normalize(0.5, 0.5)(
            transforms.ToTensor()(
                draw_single_char(ch, font, args.canvas_size)
            )
        ).unsqueeze(dim=0) for ch in src]

        label_list = [args.label for _ in src]

        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor(label_list)

        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    else:
        val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'),
                                     input_nc=args.input_nc,
                                     start_from=args.start_from)
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    global_steps = 0
    with open(args.type_file, 'r', encoding='utf-8') as fp:
        fonts = [s.strip() for s in fp.readlines()]
    writer_dict = {v: k for k, v in enumerate(fonts)}

    for batch in dataloader:
        if args.run_all_label:
            # global writer_dict
            writer_dict_inv = {v: k for k, v in writer_dict.items()}
            for label_idx in range(29):
                model.set_input(torch.ones_like(batch[0]) * label_idx, batch[2], batch[1])
                model.forward()
                tensor_to_plot = torch.cat([model.fake_B, model.real_B], 3)
                # img = vutils.make_grid(tensor_to_plot)
                save_image(tensor_to_plot, os.path.join(infer_dir, "infer_{}".format(writer_dict_inv[label_idx]) + "_construct.png"))
        else:
            infer_image, real_image = model.web_demo_sample(batch, infer_dir, result_folder, infer_name, gen_no)
            global_steps += 1
            # print(batch[0]) # yf: the label --- batch[0]

    t_finish = time.time()

    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))

    return infer_image, real_image

def web_demo_with_handwriting_img(args, handwriting_img):

    # python infer_interpolate_yf.py 
    # --experiment_dir experiment_dir 
    # --gpu_ids cuda:0 --batch_size 128 
    # --resume 90000 
    # --from_txt 
    # --src_font ./fonts/target/all_he_ttfs/998_方正瘦金书_GBK.ttf 
    # --src_txt 深 --label 17 --result_folder 20230514 --gen_no 0 --infer_name 0

    # args = parser.parse_args()
    data_dir = os.path.join(args.experiment_dir, "useful_functions/data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    chk_mkdir(infer_dir)

    result_folder = args.result_folder
    infer_name = args.infer_name
    gen_no = args.gen_no
    style = torch.zeros(1, 128)

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    t0 = time.time()

    model = Zi2ZiModelDemo(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    t1 = time.time()

    if args.from_txt:
        src = args.src_txt
        font = ImageFont.truetype(args.src_font, size=args.char_size)

        # author codes
        # au_img = draw_single_char(src, font, args.canvas_size)
        # print(f'au_img: {au_img.size}')
        # print(f'handwriting_img: {handwriting_img.shape}')
        # print(handwriting_img.shape)
        # img_list = [transforms.Normalize(0.5    , 0.5)(
        #     transforms.ToTensor()(
        #         handwriting_img
        #         # draw_single_char(ch, font, args.canvas_size)
        #     )).unsqueeze(dim=0) ]
        # for ch in src]
        def normalize_image(image):
            reshaped_tensor = image.unsqueeze(0)
            normalize = transforms.Normalize(0.5, 0.5)
            normalized_image = normalize(reshaped_tensor)
            return normalized_image

        def tensorize_image(image):
            resize = transforms.Resize((256, 256))
            
            resized_image = resize(image)
            
            return resized_image.unsqueeze(0)
        
        handwriting_img = 255 - handwriting_img
        handwriting_img = torch.from_numpy(handwriting_img).float()
        normalized_img = normalize_image(handwriting_img)
        tensorized_img = tensorize_image(normalized_img)
        img_list = [tensorized_img]

        label_list = [args.label for _ in src]

        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor(label_list)

        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    else:
        val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'),
                                     input_nc=args.input_nc,
                                     start_from=args.start_from)
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    global_steps = 0
    with open(args.type_file, 'r', encoding='utf-8') as fp:
        fonts = [s.strip() for s in fp.readlines()]
    writer_dict = {v: k for k, v in enumerate(fonts)}

    for batch in dataloader:
        if args.run_all_label:
            # global writer_dict
            writer_dict_inv = {v: k for k, v in writer_dict.items()}
            for label_idx in range(29):
                model.set_input(torch.ones_like(batch[0]) * label_idx, batch[2], batch[1])
                model.forward()
                tensor_to_plot = torch.cat([model.fake_B, model.real_B], 3)
                # img = vutils.make_grid(tensor_to_plot)
                save_image(tensor_to_plot, os.path.join(infer_dir, "infer_{}".format(writer_dict_inv[label_idx]) + "_construct.png"))
        else:
            infer_image, real_image = model.web_demo_sample(batch, infer_dir, result_folder, infer_name, gen_no)
            global_steps += 1
            # print(batch[0]) # yf: the label --- batch[0]

    t_finish = time.time()

    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))

    return infer_image, real_image

if __name__ == '__main__':
    with torch.no_grad():
        img = infer_webdemo()
        # img.show()

#  python infer_web_demo.py --experiment_dir experiment_dir --gpu_ids cuda:0 --batch_size 128 --resume 90000 --from_txt --src_font ./fonts/target/all_he_ttfs/998_方正瘦金书_GBK.ttf --src_txt 深 --label 17 --result_folder 20230514 --gen_no 0 --infer_name 0
