import os
import numpy as np
import torch
from os.path import join as pjoin
from models.vqvae import HumanVQVAE
from options.option_vq import get_args_parser
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.momask_get_opt import get_opt
from utils.word_vectorizer import WordVectorizer
from utils.eval_t2m import evaluation_vqvae_plus_mpjpe
from dataset import dataset_TM_eval
import warnings
warnings.filterwarnings('ignore')


def main():
    ##### ---- Args ---- #####
    args = get_args_parser()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}_eval')
    os.makedirs(args.out_dir, exist_ok=True)

    ##### ---- Eval Wrapper ---- #####
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' \
        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Word Vectorizer ---- #####
    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    ##### ---- Dataloader ---- #####
    args.nb_joints = 21 if args.dataname == 'kit' else 22
    dim_pose = 251 if args.dataname == 'kit' else 263

    val_loader = dataset_TM_eval.DATALoader(args.dataname, is_test=False, batch_size=32,
                            w_vectorizer=w_vectorizer, unit_length=2**args.down_t)

    ##### ---- Model ---- #####
    model = HumanVQVAE(args,
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)
    assert args.resume_pth is not None, "You must specify --resume_pth for evaluation"
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    model.load_state_dict(ckpt['net'] if 'net' in ckpt else ckpt['vq_model'])
    model.eval()
    model.to(args.device)

    ##### ---- Evaluation ---- #####
    fid, div, top1, top2, top3, matching, mae = [], [], [], [], [], [], []

    repeat_time = 20
    for i in range(repeat_time):
        f, d, R, m, l1 = evaluation_vqvae_plus_mpjpe(
            val_loader, model, i,
            eval_wrapper=eval_wrapper,
            num_joint=args.nb_joints
        )
        fid.append(f)
        div.append(d)
        top1.append(R[0])
        top2.append(R[1])
        top3.append(R[2])
        matching.append(m)
        mae.append(l1)

    # 汇总结果
    def report(name, array):
        mean = np.mean(array)
        conf = np.std(array) * 1.96 / np.sqrt(repeat_time)
        return f"{name}: {mean:.3f}, conf. {conf:.3f}"

    print("======= Final Evaluation =======")
    print(report("FID", fid))
    print(report("Diversity", div))
    print(report("TOP1", top1))
    print(report("TOP2", top2))
    print(report("TOP3", top3))
    print(report("Matching", matching))
    print(report("MAE", mae))

    # 输出到日志文件
    with open(os.path.join(args.out_dir, 'final_result.log'), 'w') as f:
        f.write("======= Final Evaluation =======\n")
        f.write(report("FID", fid) + '\n')
        f.write(report("Diversity", div) + '\n')
        f.write(report("TOP1", top1) + '\n')
        f.write(report("TOP2", top2) + '\n')
        f.write(report("TOP3", top3) + '\n')
        f.write(report("Matching", matching) + '\n')
        f.write(report("MAE", mae) + '\n')


if __name__ == '__main__':
    main()
