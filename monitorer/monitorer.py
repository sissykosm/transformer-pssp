from subprocess import run, PIPE, Popen, TimeoutExpired
import argparse
import os

scriptName = "./train.py"

batch_size = 10

d_modelArray = [32, 64, 128, 256, 512]
n_layersArray = [1, 2, 3, 4, 5, 6]
dropoutArray = [0.01, 0.1, 0.6]
attentionHeadsArray = [8]

d_modelArray.reverse()
n_layersArray.reverse()
dropoutArray.reverse()
attentionHeadsArray.reverse()

def createArgs(
    batch_size=None,
    d_model=None,
    d_k=None,
    d_v=None,
    dropout=None,
    epoch=None,
    label_smoothing=None,
    log=None,
    n_head=None,
    n_layers=None,
    n_warmup_steps=None,
    d_inner_hid=None,
    data=None,
    save_model=None,
    save_plot=None
): 
    base = ["python3", scriptName]

    if batch_size != None:
        base += ["-batch_size", str(batch_size)]

    if d_model != None:
        base += ["-d_model", str(d_model)]

    if d_k != None:
        base += ["-d_k", str(d_k)]

    if d_v != None:
        base += ["-d_v", str(d_v)]

    if dropout != None:
        base += ["-dropout", str(dropout)]

    if epoch != None:
        base += ["-epoch", str(epoch)]

    if label_smoothing != None:
        base += ["-label_smoothing", str(label_smoothing)]

    if log != None:
        base += ["-log", str(log)]

    if n_head != None:
        base += ["-n_head", str(n_head)]
    
    if n_layers != None:
        base += ["-n_layers", str(n_layers)]

    if n_warmup_steps != None:
        base += ["-n_warmup_steps", str(n_warmup_steps)]

    if d_inner_hid != None:
        base += ["-d_inner_hid", str(d_inner_hid)]
    
    if data != None:
        base += ["-data", data]

    if save_model != None:
        base += ["-save_model", save_model]

    if save_plot != None:
        base += ["-save_plot", save_plot]

    return base

def main():
    parser = argparse.ArgumentParser()
    # To make the input integers
    parser.add_argument('--only', nargs='+', type=int, default=[])
    opt = parser.parse_args()

    totalProcs = len(d_modelArray) * len(n_layersArray) * len(dropoutArray) * len(attentionHeadsArray)
    procedures = []
    procNum = 1
    
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    for d_model in d_modelArray:
        for n_layers in n_layersArray:
            for dropout in dropoutArray:
                if len(opt.only) > 0:
                    if procNum not in opt.only:
                        procNum += 1
                        continue

                args = createArgs(
                    batch_size=batch_size,
                    d_model=d_model,
                    d_v=int(d_model/4),
                    d_k=int(d_model/4),
                    n_layers=n_layers, 
                    dropout=dropout,
                    d_inner_hid=int(d_model*4),
                    log="./logs/log" + str(procNum),
                    save_model="./models/model" + str(procNum),
                    save_plot="./plots/plot" + str(procNum),
                    n_head=8,
                    epoch=200
                )

                print(args)

                proc = Popen(args)
                procedures.append(proc)
                
                print("[MONITORER] Started neural network " + str(procNum) + "/" + str(totalProcs))

                returnCode = proc.wait()
                if returnCode != 0:
                    print("[MONITORER] Oops! Something broke!")

                procNum += 1

    print("Spawned " + str(len(procedures)) + " processes.")

if __name__ == "__main__":
    main()