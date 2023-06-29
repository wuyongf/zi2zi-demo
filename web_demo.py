import gradio as gr
import cv2
from infer_web_demo import infer_webdemo, web_demo_with_handwriting_img
import torch
import argparse
import pathlib
from ttf2img.font2img_single import generate_font_image

def to_black(image):
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return output

def get_target_chara(src_font_idx, chara):
    data_root = data_root = pathlib.Path('./ttf2img/ttf_folder')
    all_image_paths = list(data_root.glob('*.ttf*'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    # print(len(all_image_paths))
    # for i in range (len(all_image_paths)):
    #     print(all_image_paths[i])
    return generate_font_image(all_image_paths[src_font_idx], f'{chara}', img_size= 256, chara_size= 192)

def to_infer(char, label):
    print(label)
    print(type(label))
    parser = argparse.ArgumentParser(description='Infer')

    parser.add_argument('--resume', type=int, default=50000)
    parser.add_argument('--src_font', default = './fonts/source/仓耳今楷03-W03.ttf')
    parser.add_argument('--src_txt', default = f'{char}')
    parser.add_argument('--label', type=int, default=f'{label}')

    parser.add_argument('--experiment_dir', default= './experiment_dir')
    parser.add_argument('--gpu_ids', default=['cuda:0'], nargs='+', help="GPUs")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--from_txt', default = 'True',action='store_true')
    parser.add_argument('--result_folder', default = '20230514')
    parser.add_argument('--gen_no', type=int, default = 0)
    parser.add_argument('--infer_name', default = '0')
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=256,help="size of your input and output image")
    parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
    parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
    # parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
    parser.add_argument('--Lcategory_penalty', type=float, default=1.0,help='weight for category loss')
    parser.add_argument('--embedding_num', type=int, default=40,help="number for distinct embeddings")
    parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--obj_path', type=str, default='./experiment/data/val.obj', help='the obj file you infer')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--canvas_size', type=int, default=256)
    parser.add_argument('--char_size', type=int, default=256)
    parser.add_argument('--run_all_label', action='store_true')
    parser.add_argument('--type_file', type=str, default='type/宋黑类字符集.txt')
        
    args = parser.parse_args()
    with torch.no_grad():
        fake_img, real_img = infer_webdemo(args)
        
        target_img = get_target_chara(label, f'{char}')
        # if(sketch_img is not None):
        #     fake_img, real_img = web_demo_with_handwriting_img(args, sketch_img)
        return fake_img, target_img, real_img 

def clear_input():
    return "", ""

# interface = gr.Interface(
#     fn=to_infer, 
#     inputs= [   gr.Textbox(lines=3, placeholder="Input One Chinese Character Here...",label="input"),
#                 gr.Dropdown(
#                     ["楷體", "柳公權體", "黃庭堅", '米芾', '張即之',
#                     "蘇軾", "孫過庭", "王獻之", '褚遂良', '趙孟頫',
#                     "魏碑", "康熙字典體", "郭沫若", '管峻', '田英章',
#                     "孫萬民", "隸書", "蘇孝慈碑", '瘦金書', '於右任',
#                     '歐陽詢'], 
#                     label="Font Style(字體)", info="Will add more styles later!", type = 'index'),
#                 # gr.Sketchpad(shape=(1024,1024)),
#             ],
#     outputs=[gr.Image(label='Imitation Image'),
#              gr.Image(label='Source Image')]
# )
# interface.launch(share=True)

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("zi2zi"):
            with gr.Row():
                with gr.Column(scale=2, min_width=600):
                    zi2zi_text = gr.Textbox(lines=3, placeholder="Input One Chinese Character Here...",label="input")
                    zi2zi_label = gr.Dropdown(
                            ["楷體", "柳公權體", "黃庭堅", '米芾', '張即之',
                            "蘇軾", "孫過庭", "王獻之", '褚遂良', '趙孟頫',
                            "魏碑", "康熙字典體", "郭沫若", '管峻', '田英章',
                            "孫萬民", "隸書", "蘇孝慈碑", '瘦金書', '於右任',
                            '歐陽詢'], 
                            label="Font Style(字體)", info="Will add more styles later!", type = 'index')
                    with gr.Row():
                        zi2zi_reset_button = gr.Button("Reset")
                        zi2zi_infer_button = gr.Button("Infer")
                zi2zi_dataframe = gr.Dataframe(
                    label='Examples',
                    headers=["Seen Character", "Unseen Character"],
                    datatype=["str", "str"],
                    row_count=2,
                    col_count=(2, "fixed"),
                    value=[
                        ["楷", "隸"],
                        ["魏", "芾"],
                        ["於", "熙"],
                        ["之", "頫"],
                    ],
                    # type='numpy'  # Set the type to 'numpy' or 'pandas' or 'array'
                )
            with gr.Row():
                zi2zi_source_img = gr.Image(label='Source Image')
                zi2zi_target_img = gr.Image(label='Target Image')
                zi2zi_fake_img = gr.Image(label='Imitation Image')

        with gr.TabItem("dg-font"):
            with gr.Row():
                with gr.Column(scale=2, min_width=600):
                    text1 = gr.Textbox(lines=3, placeholder="Input One Chinese Character Here...",label="input")
                    label = gr.Dropdown(
                            ["楷體", "柳公權體", "黃庭堅", '米芾', '張即之',
                            "蘇軾", "孫過庭", "王獻之", '褚遂良', '趙孟頫',
                            "魏碑", "康熙字典體", "郭沫若", '管峻', '田英章',
                            "孫萬民", "隸書", "蘇孝慈碑", '瘦金書', '於右任',
                            '歐陽詢'], 
                            label="Font Style(字體)", info="Will add more styles later!", type = 'index')
                    with gr.Row():
                        reset_button = gr.Button("Reset")
                        infer_button = gr.Button("Infer")
                table1 = gr.Textbox(label="t1")
            with gr.Row():
                real_img = gr.Image(label='Source Image')
                fake_img = gr.Image(label='Imitation Image')

        with gr.TabItem("fs-font"):
            with gr.Row():
                with gr.Column(scale=2, min_width=600):
                    text1 = gr.Textbox(lines=3, placeholder="Input One Chinese Character Here...",label="input")
                    label = gr.Dropdown(
                            ["楷體", "柳公權體", "黃庭堅", '米芾', '張即之',
                            "蘇軾", "孫過庭", "王獻之", '褚遂良', '趙孟頫',
                            "魏碑", "康熙字典體", "郭沫若", '管峻', '田英章',
                            "孫萬民", "隸書", "蘇孝慈碑", '瘦金書', '於右任',
                            '歐陽詢'], 
                            label="Font Style(字體)", info="Will add more styles later!", type = 'index')
                    with gr.Row():
                        reset_button = gr.Button("Reset")
                        infer_button = gr.Button("Infer")
                table1 = gr.Textbox(label="t1")
            with gr.Row():
                real_img = gr.Image(label='Source Image')
                fake_img = gr.Image(label='Imitation Image')

    zi2zi_infer_button.click(to_infer, inputs=[zi2zi_text, zi2zi_label], outputs=[zi2zi_fake_img, zi2zi_target_img, zi2zi_source_img])
    zi2zi_reset_button.click(clear_input, inputs=[], outputs=[zi2zi_fake_img, zi2zi_source_img])
demo.launch(share=True)

