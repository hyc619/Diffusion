import argparse
import os, json, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import stanza
import spacy_stanza
import numpy as np
import torch as th
from transformers import set_seed
import torch.distributed as dist
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer
from improved_diffusion.test_util import get_weights, denoised_fn_round
from functools import partial
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
sys.path.insert(0, '/data/hanyc/project/Diffusion-LM-main/transformers/examples/pytorch/language-modeling')
from custom_trainer import Classifier_GPT2, Classifier_Times, Classifier_POS, Classifier_Tree
from infill_util import langevin_fn3, get_score, langevin_fn3_compose, langevin_fn1, langevin_fn4, langevin_fn_tree, langevin_fn_length
from spacy.lang.en import English
import time
from tqdm import *





def main():
    set_seed(101)
    args = create_argparser().parse_args()

    # load configurations.
    print(os.path.split(args.model_path))

    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    args.__dict__.update(training_args)

    args.noise_level = 0.0
    args.sigma_small = True

    args.diffusion_steps = 200  # 500  # DEBUG
    dist_util.setup_dist()
    logger.configure()
    print(args.clip_denoised, 'clip_denoised')
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path))
    model.to(dist_util.dev())
    model.eval()

    logger.log("load embedding models")
    print(os.path.split(args.model_path)[0])
    model_embs, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                   os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        print('e2e, load the right model embeddings', '*'*80)
        model_embs.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_embs = model_embs.cuda()
    model3 = get_weights(model_embs, args)

    logger.log('load the partial sequences')

    partial_seq = ['A kid friendly venue named Alimentum is located on the riverside .',
                   'Alimentum , situated by the river , is quite child friendly .']
    partial_seq_idx = ['0', '1']

    tokens2id = {v: k for k, v in tokenizer.items()}
    todo_pad_token = -1
    pad_token = tokens2id['PAD']
    encoded_partial_seq = [th.LongTensor([tokens2id.get(x, tokens2id['UNK']) for x in seq.split()]) for seq in
                           partial_seq]

    right_pad = th.empty(64).fill_(pad_token).long()
    encoded_partial_seq = [th.cat([right_pad], dim=0)]
    encoded_partial_seq[0][0] = tokens2id['START']
    encoded_partial_seq[0][args.tgt_len] = tokens2id['END']

    control_label_lst = []
    with open('/data/hanyc/project/Diffusion-LM-main/datasets/control_target/target_attribute.json',
              'r') as controlf:
        for line in controlf:
            control_label_lst.append(json.loads(line))
    # print(control_label_lst[:5])


    logger.log("sampling...")
    sample_dict = {}

    for lable in tqdm(control_label_lst, total=len(control_label_lst)):
        control = [tokens2id.get(x, tokens2id['UNK']) for x in lable]

        all_images = []
        all_labels = []
        seqlen = 64 - len(control)
        # print(args.in_channel)
        sample_shape = (1, seqlen, args.in_channel)
        randm = th.randn(sample_shape).cuda()
        # print('randm', randm.shape)
        prompt = th.tensor(control)
        # print('prompt', prompt)
        p_emb = model3(prompt.cuda())
        p_emb = p_emb.unsqueeze(0)
        # print('p_emb', p_emb.shape)


        noise = th.cat((p_emb, randm), dim=1)

        # print('noise', noise)
        sample_shape = noise.shape
        # print('sample_shape', sample_shape)

        loop_func_ = diffusion.p_sample_loop_progressive
        for sample in loop_func_(
                model,
                sample_shape,
                noise=noise,
                denoised_fn=partial(denoised_fn_round, args, model3.cuda()),
                # denoised_fn=partial(langevin_early, model_control, model3.cuda(),
                #                     label_ids.expand(args.batch_size, -1), 0.1),
                clip_denoised=args.clip_denoised,
                model_kwargs=None,
                device=None,
                # langevin_func=partial(langevin_func, model_control,
                #                       label_ids.expand(args.batch_size, -1), 0.01),
        ):
            final = sample["sample"]

        sample = final

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]

        label = ''
        for i in lable:
            label = label + ' ' + i




        sample_dict[label] = arr
        print(f'writing to sample_dict, for class {" ".join(label)}')




    def decode_helper(args, sample_dict, diff_model=None):
        result_dict = {}
        if not diffusion.training_mode.startswith('e2e'):
            logger.log('decode by rounding. ')
            set_seed(101)
            model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path,
                                           args.in_channel,
                                           os.path.split(args.model_path)[0])

        for k, v in sample_dict.items():

            arr = v
            if diffusion.training_mode.startswith('e2e'):
                word_lst_e2e = []
                print('decoding for e2e', )
                x_t = th.tensor(arr).cuda()
                print(x_t.shape)
                if args.model_arch == 'conv-unet':
                    reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
                else:
                    reshaped_x_t = x_t
                logits = diff_model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
                cands = th.topk(logits, k=1, dim=-1)
                tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
                for seq in cands.indices:
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                    word_lst_e2e.append(tokens)
                word_lst = word_lst_e2e
            else:
                word_lst = rounding_func(args.experiment, arr, model, tokenizer)
            result_dict[k] = word_lst
        return result_dict

    print(f'sampled for {len(sample_dict)} control tasks')
    # out_path_pipe = os.path.join(args.out_dir, f"{model_base_name}.infill_{args.eval_task_}_{args.notes}.json")
    model_base_name = os.path.basename(
        os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_path_pipe = os.path.join('/data/hanyc/project/Diffusion-LM-main/improved-diffusion/out_gen/',
                                 f"prompt.json")
    fout = open(out_path_pipe, 'w')
    result_dict = decode_helper(args, sample_dict, diff_model=model)
    for k, word_lst in result_dict.items():
        print({k: word_lst}, file=fout)
    fout.close()
    print(f'written the decoded output to {out_path_pipe}')
    out_path2 = out_path_pipe

    args.out_path2 = out_path2
    return args









def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=False, use_ddim=False, eta=1.0, num_samples=50, batch_size=1, model_path="/data/hanyc/project/Diffusion-LM-main/improved-diffusion/diffusion_models/diff_e2e-tgt_block_rand16_transformer_lr0.0001_0.0_2000_sqrt_Lsimple_h128_s2_d0.1_sd102_conditionprompt_xstart_e2e/model200000.pt",
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        emb_scale_factor=1.0, split='train', debug_path='', eval_task_='infill',
        partial_seq="", partial_seq_file="", verbose='yes', tgt_len=15, t_merge=200, interp_coef=0.5, notes='',
        start_idx=0, end_idx=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    args = main()