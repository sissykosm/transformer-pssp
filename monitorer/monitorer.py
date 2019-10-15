from subprocess import run, PIPE, Popen, TimeoutExpired

scriptName = "./monitorer/mock-process.py"

batch_size = 5

d_modelArray = [32, 64, 128, 256, 512]
n_layersArray = [1, 2, 3, 4, 5, 6]
dropoutArray = [0.01, 0.1, 0.6]

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
    data=None
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

    return base

def main():
    totalProcs = len(d_modelArray) * len(n_layersArray) * len(dropoutArray)
    procedures = []
    procNum = 1

    for d_model in d_modelArray:
        for n_layers in n_layersArray:
            for dropout in dropoutArray:
                args = createArgs(
                    batch_size=batch_size,
                    d_model=d_model,
                    d_v=int(d_model/4),
                    d_k=int(d_model/4),
                    n_layers=n_layers, 
                    dropout=dropout,
                    d_inner_hid=int(d_model*4),
                    log="log" + str(procNum)
                )

                print(args)

                proc = Popen(args)
                procedures.append(proc)
                
                print("[MONITORER] Started neural network " + str(procNum) + "/" + str(totalProcs))

                proc.wait()
                procNum += 1

    print("Spawned " + str(len(procedures)) + " processes.")

if __name__ == "__main__":
    main()