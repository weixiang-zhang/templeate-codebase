import os
from util import misc
from util.tensorboard import writer
from util.logger import log
from util.recorder import recorder
from opt import Opt

from inr_trainer.train import train_inr_set



def main():
    opt = Opt()
    args = opt.get_args()
    misc.fix_seed(args.seed)
    log.inst.success("start")
    writer.init_path(args.save_folder)
    log.set_export(args.save_folder)

    ###################################################################################################
    train_inr_set(args)
    ###################################################################################################
    time_dic = log.end_all_timer()

    table = recorder.add_main_table()
    if table:
        # recorder.add_summary_table()
        recorder.dic["time"] = time_dic
        recorder.add_time_table()
        recorder.dump_table(os.path.join(args.save_folder, "res_table.md"))

    writer.close()
    log.inst.success("Done")



if __name__ == "__main__":
    main()
