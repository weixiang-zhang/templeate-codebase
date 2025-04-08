import os

def inr_demo(use_cuda=0):
    '''Args for EVOS(stepwise scheduler) with only one image'''
    args = [
        "--model_type",
        "siren",
        "--eval_lpips",
        "--log_epoch",
        "500",
        "--num_epochs",
        "500",
        "--tag",
        "demo",
        "--lr",
        "0.0001",
        "--up_folder_name",
        "000_demo",
        # "--input_path",
        # "./data/div2k/test_data/00.png",

        "--multi_data",
        "--input_path",
        "./data/div2k/test_data",

    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_cuda) 
    script = "python main.py " + " ".join(args)
    print(f"Running: {script}")
    os.system(script)


if __name__ == "__main__":
    inr_demo(0)
