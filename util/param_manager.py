import os
import subprocess
from types import SimpleNamespace
import math
from config import CONFIG


def run_subprocess(param_list, gpu_list, exp_num):
    processes = []

    # assert len(param_list) == len(gpu_list)
    _len = min(len(param_list), len(gpu_list))
    param_list = param_list[:_len]
    gpu_list = gpu_list[:_len]

    for param, use_cuda in zip(param_list, gpu_list):
        pm = ParamManager(param=param, exp_num=exp_num)
        cmd_str = pm.export_cmd_str(use_cuda=[use_cuda])
        ###### print cmd str for debugger ######
        # exit()
        ########################################
        process = subprocess.Popen(cmd_str, shell=True)
        print(f"PID: {process.pid}")
        processes.append(process)

    for process in processes:
        process.wait()


def run_tasks(exp_num, param_list, gpu_list):

    gpus = len(gpu_list)
    rounds = math.ceil(len(param_list) / gpus)
    print("rounds: ", rounds)

    for i in range(rounds):
        cur_param_list = param_list[i * gpus : min(len(param_list), (i + 1) * gpus)]
        cur_len = len(param_list)
        gpu_list = gpu_list[:cur_len]
        run_subprocess(cur_param_list, gpu_list, exp_num)



class ParamManager(object):
    def __init__(self, **kw):
        
        self.p = SimpleNamespace()
        self._set_exp(**kw)

    def _set_default_parmas(self):
        self.p.model_type = "siren"
        self.p.eval_lpips = CONFIG.STORE_TRUE
        self._use_single_data()

    def _set_exp(self, param="", exp_num="000"):
        self.exp_num = exp_num
        self._exp_name = f"exp_{exp_num}"
        self._set_default_parmas()
        self.p.tag = param
        eval(f"self._set_exp_{exp_num}(param)")
        self.p.lr = self._get_lr_by_model(self.p.model_type)
        self.p.up_folder_name = self._exp_name

    def _convert_dict_args_list(self):
        args_dic = vars(self.p)
        args_list = []
        for key, val in args_dic.items():
            args_list.append(f"--{key}")
            if val is not CONFIG.STORE_TRUE:
                args_list.append(str(val))
        self._print_args_list(args_list)
        return args_list

    def export_args_list(self):
        return self._convert_dict_args_list()

    def export_cmd_str(self, use_cuda=[0]):
        args_list = self._convert_dict_args_list()
        script = "python main.py " + " ".join(args_list)
        script = self.add_cuda_visible_to_script(script, use_cuda)
        return script

    @staticmethod
    def add_cuda_visible_to_script(script, use_cuda=[0]):
        visible_devices: str = ",".join(map(str, use_cuda))
        return f"CUDA_VISIBLE_DEVICES={visible_devices} {script}"

    @staticmethod
    def _print_args_list(args_list):
        print("#" * 10 + "print for vscode debugger" + "#" * 10)
        for item in args_list:
            print(f'"{item}",')

    def _get_lr_by_model(self, model):
        if model == "gauss" or model == "wire":
            return 5e-3
        elif model == "siren":
            return 1e-4  # 1e-4 | 5e-4
        elif model == "finer":
            return 5e-4
        elif model == "pemlp":
            return 1e-3
        else:
            raise NotImplementedError

    def _use_single_data(self, pic_index="00", datasets = CONFIG.DIV2K_TEST):
        if hasattr(self.p, "multi_data"):
            delattr(self.p, "multi_data")

        self.p.input_path = os.path.join(datasets, f"{pic_index}.png")
        self._exp_name += f"_single_{pic_index}"

    def _use_datasets(self, dataset=CONFIG.DIV2K_TEST):
        self.p.multi_data = CONFIG.STORE_TRUE
        self.p.input_path = dataset
        dataset_name = os.path.basename(dataset)
        self._exp_name += f"_{dataset_name}"


    def _set_exp_001(self, param):
        
        ####  update it if you want ... 
        # self.p.tag = param
        self._exp_name += "_50_epoch"

        self.p.log_epoch = 50 
        self.p.num_epochs = 50

        self.p.model = param

        # self._use_single_data("00")
        self._use_datasets()



     