import os
from pathlib import Path
import argparse

from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser()
parser.add_argument('--sample_count', type=int, default=200, help='sample_count')
parser.add_argument('--src_font', type=str, default='仓耳今楷03-W03.ttf', help='source font')

args = parser.parse_args()

def get_char_list_from_ttf_without_origin(font_file,original_char_keys):
    ' 给定font_file,获取它的中文字符 '
    f_obj = TTFont(font_file)
    m_dict = f_obj.getBestCmap()

    unicode_list = []

    for key, _ in m_dict.items():
        # 中日韩统一表意文字 范围: 4E00—9FFF // CJK Unified Ideographs. Range: 4E00—9FFF
        if key >= 0x4E00 and key <= 0x9FFF:
            if hex(key) in original_char_keys:
                # print('get!!')
                pass
            unicode_list.append(key)

    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    return char_list

def get_char_list_from_ttf(font_file):
    ' 给定font_file,获取它的中文字符 '
    f_obj = TTFont(font_file)
    m_dict = f_obj.getBestCmap()

    unicode_list = []

    count = 0
    for key, _ in m_dict.items():
        # 中日韩统一表意文字 范围: 4E00—9FFF // CJK Unified Ideographs. Range: 4E00—9FFF
        if key >= 0x4E00 and key <= 0x9FFF:
            count+=1
            unicode_list.append(key)
            if key == 0x548C:
                print(count)

    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    return char_list

def ttf2imgs(font_label, sample_count):

    sample_label = font_label

    font_file_name =  writer_dict_inv[sample_label]

    font_file_addr = './fonts/target/'

    font_file = font_file_addr + font_file_name 

    chars = get_char_list_from_ttf(font_file)
    print(chars)

    npchars_unsort1 = np.array(chars)
    npchars = npchars_unsort1[None, :]

    print(npchars.shape)
    shape = npchars.shape[1]
    # print(npchars[0][1]) # for index

    # Write .txt files
    npchars_addr = "./fonts/target_char/"
    os.makedirs(npchars_addr, exist_ok=True)

    exist_char_name = '已有字_' + str(shape) + '_' + font_file_name + '.txt'
    np.savetxt(npchars_addr + exist_char_name, npchars, delimiter="", fmt="%s", encoding='utf-8')

    # init
    dst_font_addr = font_file
    src_font_addr = f"./fonts/source/{args.src_font}"
    char_set_addr = npchars_addr + exist_char_name

    # command
    os.system(
        "python font2img.py --src_font=" + src_font_addr + " --dst_font=" + dst_font_addr + " --charset=" + char_set_addr +
        " --sample_count=" + str(sample_count) + " --sample_dir=sample_dir --label=" + str(
            sample_label) + " --filter --shuffle --mode=font2font")

def ttf2imgs_mix_original(font_label, sample_count, original_path):
    # Init
    # sample_count = 2000
    sample_label = font_label

    font_file_name =  writer_dict_inv[sample_label]

    font_file_addr = './fonts/target/'
    font_file_format = ".ttf"
    # font_file_format = ".otf"

    font_file = font_file_addr + font_file_name + font_file_format

    # get the original chars
    original_chars = []
    original_char_keys = []
    for jpg_file in os.listdir(original_path):
        each_char = Path(jpg_file).stem
        original_chars.append(each_char)

    for original_char in original_chars:
        print(original_char)
        original_char_key = hex(ord(original_char))
        original_char_keys.append(original_char_key)

    count = len(original_chars)

    chars = get_char_list_from_ttf_without_origin(font_file, original_char_keys)
    print(chars)

    npchars_unsort1 = np.array(chars)
    npchars = npchars_unsort1[None, :]

    print(npchars.shape)
    shape = npchars.shape[1]
    # print(npchars[0][1]) # for index

    # Write .txt files
    npchars_addr = "./fonts/target_exist_char/"
    os.makedirs(npchars_addr, exist_ok=True)

    exist_char_name = '已有字_' + str(shape) + '_' + font_file_name + '.txt'
    np.savetxt(npchars_addr + exist_char_name, npchars, delimiter="", fmt="%s", encoding='utf-8')

    # init
    dst_font_addr = font_file
    src_font_addr = f"./fonts/source/{args.src_font}"
    char_set_addr = npchars_addr + exist_char_name

    # command
    os.system(
        "python font2img_yf.py --src_font=" + src_font_addr + " --dst_font=" + dst_font_addr + " --charset=" + char_set_addr +
        " --sample_count=" + str(sample_count) + " --sample_dir=sample_dir --label=" + str(
            sample_label) + " --filter --shuffle --mode=font2font" + " --count=" + str(count))

def get_target_char_list():
    directory = "./fonts/target"  # Replace with the actual directory path

    # Get all files in the directory
    files = [file for file in os.listdir(directory) if file.lower().endswith((".ttf", ".otf"))]

    # Sort the files
    files.sort()

    # Generate the dictionary
    writer_dict = {file: i for i, file in enumerate(files)}

    # Write the file names and indexes to target_list.txt
    with open("./fonts/target_list.txt", "w") as file_handle:
        for file_name, index in writer_dict.items():
            file_handle.write(f"{index}: {file_name}\n")

    print("target_list.txt generated successfully.")
    return writer_dict
    pass

writer_dict = get_target_char_list()

# writer_dict = {
#         '仓耳今楷03-W03.ttf': 0,'柳公权柳体.ttf': 1, '黄庭坚书法字体.ttf': 2, '米芾体.ttf': 3, '汉仪新蒂张即之体.ttf': 4, 
#         '书体坊苏轼行书.ttf': 5, '孙过庭草书.ttf': 6, '方正王献之小楷简.ttf': 7, '方正褚遂良楷书简.ttf': 8, '方正赵孟頫楷书简繁.ttf': 9, 
#         '汉仪魏碑简.ttf': 10, '康熙字典体.otf': 11, '书体坊郭沫若字体.ttf': 12, '方正字迹-管峻楷书简体.TTF': 13, '田英章毛笔楷书3500字.ttf': 14,
#         '汉仪孙万民草书繁.ttf': 15, '56号-洪亮毛笔隶书简体.ttf': 16, '汉仪新蒂苏孝慈碑.ttf': 17, '汉仪瘦金书简.TTF': 18, '书体坊于右任标准草书.TTF': 19 ,
#         '華康歐陽詢體.TTF': 20
#     }

# global writer_dict
writer_dict_inv = {v: k for k, v in writer_dict.items()}

if __name__ == '__main__':

    """
    process all labels: from 0 to 20
    """
    for label_no in range(0, len(writer_dict), 1):
        print("processing lable: ", label_no)
        ttf2imgs(label_no, args.sample_count)

    """
    generate all gen_n characters
    """
    # input_char_addr = './useful_functions/data/ArtMuseum-exp01/蘇軾/年.jpg'
    # # input_char_addr = './useful_functions/data/蘇東坡-和.jpeg'
    # # input_char_addr = './useful_functions/data/于右任-標準草書千字文-和-vector1.jpg'
    # # input_char_addr = './useful_functions/data/于右任-標準草書千字文-和-行書五言聯1.png'
    # # input_char_addr = './useful_functions/data/和-宋徽宗.jpg'

    # gen_no = 1

    # for label_no in range(0, 20, 1):
    #     img_name = "gen0_s"
    #     infer_name  = img_name + str(label_no)

    #     # command  # m1: 100000 m2: 6000
    #     os.system(
    #         "python infer_interpolate_he_demo.py --experiment_dir experiment_dir --gpu_ids cuda:0 --batch_size 32 --resume 100000 "
    #         "--from_txt --src_font ./fonts/source/仓耳今楷03-W03.ttf --src_txt 和 --label " + str(label_no) +
    #         " --result_folder 20221220_testing --infer_name " + infer_name + ' --gen_no ' + str(gen_no) + ' --input_char ' + str(input_char_addr))

    '''
    For 100个和字  --> Input 真跡 --> Existing Model --> Get the result
    '''
    # img_path = './useful_functions/data/100個和字/'
    # for jpg_file in os.listdir(img_path):
    #     input_char_addr = img_path + jpg_file
    #
    #     # # rename the files
    #     # r = jpg_file.replace(" ", "")
    #     # r1 = r.replace("-", "_")
    #     # if (r != jpg_file):
    #     #     os.rename(os.path.join(img_path, jpg_file) , os.path.join(img_path, r1))
    #
    #     save_folder_name = Path(input_char_addr).stem
    #     save_addr = './experiment_dir/infer/20221220_testing/100和to20Styles/' + save_folder_name
    #     if not os.path.exists(save_addr):
    #         os.makedirs(save_addr)
    #     save_addr1 = '20221220_testing/100和to20Styles/' + save_folder_name
    #
    #     gen_no = 0
    #
    #     for label_no in range(0, 20, 1):
    #         img_name = "gen0_s"
    #         infer_name  = img_name + str(label_no)
    #
    #         # command
    #         os.system(
    #             "python infer_interpolate_he_demo.py --experiment_dir experiment_dir --gpu_ids cuda:0 --batch_size 32 --resume 100000 "
    #             "--from_txt --src_font ./fonts/source/仓耳今楷03-W03.ttf --src_txt 和 --label " + str(label_no) +
    #             " --result_folder " + save_addr1 + "/ --infer_name " + infer_name + ' --gen_no ' + str(gen_no) + ' --input_char ' + str(input_char_addr))

    """
    crop the image
    """
    #
    # for i in range(0, 20, 1):
    #     img_name = "gen0_s"
    #     infer_name = img_name + str(i)
    #
    #     images_addr = "./experiment_dir/infer/20221208_results/"
    #     img_name = images_addr + infer_name + ".png"
    #
    #     head = Image.open(img_name)
    #     headbox = (0, 0, 256, 256)
    #
    #     head.crop(headbox).save(images_addr + infer_name + ".png")




