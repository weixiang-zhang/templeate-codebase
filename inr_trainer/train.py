import copy
import os
from inr_trainer.img_trainer import ImageTrainer

def train_single_inr(args):
    trainer = ImageTrainer(args)
    trainer.train()
    res = getattr(trainer, "result", None)
    return res


def train_inr_set(args):
    if args.multi_data:
        assert os.path.isdir(args.input_path)
        dir = args.input_path
        entries = os.listdir(dir)
        files = [
            entry for entry in entries if os.path.isfile(os.path.join(dir, entry))
        ]
        samples = sorted(files, key=lambda x: int(x.split(".")[0]))
        process_task(samples, args, cuda_num=0)

    else:
        assert os.path.isfile(args.input_path)
        train_single_inr(args)

def process_task(sample_list, args, cuda_num=0):
    results = []
    for sample in sample_list:
        cur_args = copy.deepcopy(args)
        cur_args.device = f"cuda:{cuda_num}"
        cur_args.input_path = os.path.join(args.input_path, sample)
        cur_res = train_single_inr(cur_args)
        results.append(cur_res)
    return results
