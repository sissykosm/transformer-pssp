from subprocess import run, PIPE, Popen, TimeoutExpired
import os

scriptName = "./translate.py"

batch_size = 5

modelArray = [
    ['models', 89],
    ['models', 50],
    ['models', 36],
    ['models', 72],
    ['models', 71],
    ['models', 54],
    ['models', 53],
    ['models', 18],
    ['models-ls', 89],
    ['models-ls', 50],
    ['models-ls', 18],
    ['models-ls', 35],
    ['models-ls', 36],
    ['models-ls', 54],
    ['models-ls', 53],
    ['models-ls', 52],
    ['models-ls', 71],
    ['models-16atth', 90],
    ['models-16atth', 89],
    ['models-16atth', 72],
    ['models-16atth', 71],
    ['models-16atth', 70],
    ['models-16atth', 54],
    ['models-16atth', 53],
    ['models-16atth', 36],
    ['models-16atth', 35],
]

def createArgs(
    batch_size=None,
    model=None,
    vocab=None,
    beam_size=None,
    output=None,
): 
    base = ["python3", scriptName]

    if batch_size != None:
        base += ["-batch_size", str(batch_size)]

    if batch_size != None:
        base += ["-beam_size", str(beam_size)]

    if model != None:
        base += ["-model", model]

    if vocab != None:
        base += ["-vocab", vocab]

    if output != None:
        base += ["-output", output]

    return base

def main():
    totalProcs = len(modelArray)
    procedures = []
    procNum = 1
    
    if not os.path.exists("./decoded_beam"):
        os.makedirs("./decoded_beam")

    for model in modelArray:
        for beam_size in [5,10]:
            [folder, number] = model
            args = createArgs(
                        batch_size=batch_size,
                        model='./' + folder + '/model' + str(number) + '.chkpt',
                        vocab="./pssp-data/data.pt",
                        beam_size=beam_size,
                        output="./decoded_beam/pred-" + folder + "-" + str(number) + "-" + str(beam_size) + ".txt"
                    )

            print(args)
            proc = Popen(args)
            procedures.append(proc)
        
            print("[MONITORER] Started decoder network " + str(procNum) + "/" + str(totalProcs))
            returnCode = proc.wait()
            if returnCode != 0:
                print("[MONITORER] Oops! Something broke!")
            procNum += 1

    print("Spawned " + str(len(procedures)) + " processes.")

if __name__ == "__main__":
    main()