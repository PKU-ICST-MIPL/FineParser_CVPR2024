import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import subprocess
import sys
def main():

    training_script = "main.py"
    world_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))  
    master_addr = "localhost"
    master_port = "14312"  
    training_script_args = ["--archs", "FineParser", "--benchmark", "FineDiving"] 
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    processes = []
    for rank in range(world_size):
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        process_args = [sys.executable, "-u", training_script] + training_script_args
        p = subprocess.Popen(process_args)
        processes.append(p)

    for p in processes:
        p.wait()

if __name__ == "__main__":
    main()
