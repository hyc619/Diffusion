import argparse
import json, torch, os
import numpy as np
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.text_datasets import load_data_control
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import pickle
from transformers import AutoTokenizer
from improved_diffusion.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from improved_diffusion.test_util import get_weights, compute_logp
from improved_diffusion.rounding import load_models, load_tokenizer
import torch.distributed as dist
import wandb
from transformers import BertTokenizer, BartForConditionalGeneration
from tqdm import *

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed)
    dist_util.setup_dist() # DEBUG **
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())  # DEBUG **
    pytorch_total_params = sum(p.numel() for p in model.parameters())  # 返回网络中的参数个数
    logger.log(f'the parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)  # 选择均匀采样或者基于loss的重要性采样

    logger.log(f'saving the hyperparameters to {args.checkpoint_path}/training_args.json')

    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "diffusion_lm"),
        name=args.checkpoint_path,
    )
    wandb.config.update(args.__dict__, allow_val_change=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.log("creating data loader...")

    if args.modality == 'BART_diffusion':
        print('load data', '*'*50)

        tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)
        bart = BartForConditionalGeneration.from_pretrained(args.pretrain_model).to(device)
        train_dir = '/data/hanyc/project/Diffusion-LM-main/datasets/valid.csv'
        adam = torch.optim.Adam(bart.parameters(), lr=args.lr)
        train_dataset = load_data_control(args, train_dir, tokenizer, bart, 'train', device)
        encoder_latent = []
        bart.eval()
        for i, cur in tqdm(enumerate(train_dataset), total=len(train_dataset)):
            cur = {k: v.to(device) for k, v in cur.items()}
            with torch.no_grad():
                out = bart(**cur)
                #print(diff_input.shape)
                adam.zero_grad()

            diff_input = out['encoder_last_hidden_state'].cpu().numpy()
            encoder_latent.append(diff_input)

        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=encoder_latent,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            checkpoint_path=args.checkpoint_path,
            gradient_clipping=args.gradient_clipping,
            eval_data=encoder_latent,
            eval_interval=args.eval_interval
        ).run_loop()









def create_argparser():
    '''从字典中自动生成命令行传参的argumen parser'''
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=101,
        gradient_clipping=-1.0,
        eval_interval=2000,
        checkpoint_path='diff_models'
    )
    text_defaults = dict(modality='BART_diffusion',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         config='diffusion_lm/synthetic_data/configs/emnlp2020/experiments/difflm_seed0_m3_k128_trainc20000.yaml',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress',model_arch='conv-unet',
                         roc_train='diffusion_lm/ROCstory',#'diffusion_lm/ROCstory/ROCstory17.csv',
                         wiki_train='diffusion_lm/simple_wiki/data.v1.split/simple.training.txt',
                         e2e_train='e2e_data',
                         yelp_train='diffusion_lm/yelpnlg-resources/yelpnlg-corpus',
                         commonGen_train = 'diffusion_lm/common-gen/commongen_data',
                         emb_scale_factor=1.0, noise_level=0.0, cache_mode='no', use_bert_tokenizer='no',
                         padding_mode='block',
                         preprocessing_num_workers=1,
                         pretrain_model='/data/hanyc/project/Diffusion-LM-main/my_code/Pretrain Model/BART_BASE_CHINESE',
                         )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()