import argparse
import torch
import numpy as np
import sys
import os
import math
import json
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset, InTheWildDataset, FakeNewsDataset, safe_collate_fn
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from PIL import Image
from editings import latent_editor
from utils.align_utils import align_face, attach_face


def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'car' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    aligner = net.grid_align
    editor = latent_editor.LatentEditor(net.decoder, is_cars)

    # load fakenews dataset
    posts = json.load(open(args.json_dir))
    

    ########################### multi thread ####################
    TOTAL = len(posts)
    THREAD_NUM = args.thread_num
    SIZE = math.ceil(TOTAL/THREAD_NUM)
    ID = args.thread_id
    START = ID * SIZE
    END = min((ID+1) * SIZE, TOTAL)
    posts = posts[START:END]
    print(f"Thread {ID}: from post {START} to post {END-1} ")
    ########################### multi thread ####################

    # split posts into sub posts and loop
    total_size = END - START
    loop_size = args.loop_size
    
    processed_num = 0
    skipped_num = 0
    # skipped_id = []
    pbar = tqdm(range(0, total_size, loop_size))
    for iter_idx in pbar:
        posts_sub = posts[iter_idx : min(iter_idx+loop_size, total_size)]
        args.image_paths = [os.path.join(args.image_root, post['image_path'][2:]) for post in posts_sub]
        args, data_loader, dataset = setup_data_loader(args, opts)

        # initial inversion
        latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)

        # set the editing operation
        if args.edit_attribute == 'inversion':
            pass
        elif args.edit_attribute == 'age' or args.edit_attribute == 'smile':
            interfacegan_directions = {
                    'age': './editings/interfacegan_directions/age.pt',
                    'smile': './editings/interfacegan_directions/smile.pt' }
            edit_direction = torch.load(interfacegan_directions[args.edit_attribute]).to(device)
        else:
            ganspace_pca = torch.load('./editings/ganspace_pca/ffhq_pca.pt') 
            ganspace_directions = {
                'eyes':            (54,  7,  8,  20),
                'beard':           (58,  7,  9,  -20),
                'lip':             (34, 10, 11,  20) }            
            edit_direction = ganspace_directions[args.edit_attribute]

        edit_directory_path = os.path.join(args.save_dir, args.edit_attribute)
        os.makedirs(edit_directory_path, exist_ok=True)

        # perform high-fidelity inversion or editing
        for i, batch in enumerate(data_loader):
            pbar.set_postfix({
                    'progress': f'{processed_num}/{total_size}',
                    'skipped': skipped_num,
                })
            
            if args.n_sample is not None and i > args.n_sample:
                print('inference finished!')
                break
            
            processed_num += 1
            if batch is None:
                skipped_num += 1
                # pbar.update()
                # skipped_id.append[posts_sub[i]['id']]
                # print(f"No face detected in id {posts_sub[i]['id']}, skip.")
                continue
            x = batch[0].to(device).float()  # images

            # calculate the distortion map
            imgs, _ = generator([latent_codes[i].unsqueeze(0).to(device)],None, input_is_latent=True, randomize_noise=False, return_latents=True)
            res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

            # produce initial editing image
            # edit_latents = editor.apply_interfacegan(latent_codes[i].to(device), interfacegan_direction, factor_range=np.linspace(-3, 3, num=40))  
            if args.edit_attribute == 'inversion':
                img_edit = imgs
                edit_latents = latent_codes[i].unsqueeze(0).to(device)
            elif args.edit_attribute == 'age' or args.edit_attribute == 'smile':
                img_edit, edit_latents = editor.apply_interfacegan(latent_codes[i].unsqueeze(0).to(device), edit_direction, factor=args.edit_degree)
            else:
                img_edit, edit_latents = editor.apply_ganspace(latent_codes[i].unsqueeze(0).to(device), ganspace_pca, [edit_direction])

            # align the distortion map
            img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=(256,256) , mode='bilinear')
            res_align  = net.grid_align(torch.cat((res, img_edit), 1))

            # consultation fusion
            conditions = net.residue(res_align)
            imgs, _ = generator([edit_latents],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
            if is_cars:
                imgs = imgs[:, :, 64:448, :]
            
            imgs = torch.nn.functional.interpolate(imgs, size=(256,256), mode='bilinear')
            
            # attach edited faces to original images ==================
            for j in range(len(imgs)):
                face_img = tensor2im(imgs[j])
                orig_img = batch[1][j]
                quad = batch[2][j].detach().cpu().numpy()
                crop = batch[3][j].detach().cpu().numpy()
                pad = batch[4][j].detach().cpu().numpy()
                edited_img = attach_face(face_img, orig_img, quad, crop, pad)
            # =========================================================

                # save images
                orig_name = os.path.basename(dataset.paths[i]).split('.')[0]
                im_save_path = os.path.join(edit_directory_path, f"{posts_sub[i]['id']}-{args.edit_attribute}.jpg")
                edited_img.save(im_save_path)


def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
   
    predictor_path = "./predictor/shape_predictor_68_face_landmarks.dat"
    align_function = align_face
    
    test_dataset = FakeNewsDataset( image_paths=args.image_paths,
                                    predictor_path = predictor_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                            #  batch_size=args.batch,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers,
                             drop_last=True,
                             collate_fn=safe_collate_fn)
                            # )

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader, test_dataset


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break

            if batch is None:
                latents = torch.zeros(1, 18, 512).to(device).float()  # if no face, return zeros. Only support batch_size=1.
            else:
                x = batch[0]  # images
                inputs = x.to(device).float()
                latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--image_root", type=str, default=None, help="The root to the image folders")
    # parser.add_argument("--images_dir", type=str, default=None, help="The directory to the images")
    parser.add_argument("--json_dir", type=str, default=None, help="The directory to the images")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
    parser.add_argument("--ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
    parser.add_argument("--edit_degree", type=float, default=0, help="edit degree")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers")
    parser.add_argument("--loop_size", type=int, default=1, help="number of posts per loop")
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--thread_id', type=int, default=0)
    
    # parser.add_argument("--gpu", type=str, default='4', help="The id of the gpu to use")

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)