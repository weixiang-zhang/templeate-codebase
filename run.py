from util.param_manager import ParamManager,run_tasks
_gpu_list = [0,1,2,3,4,5,6,7]

run_tasks("001", ["siren", "finer", "gauss", "wire"], _gpu_list)


   
    