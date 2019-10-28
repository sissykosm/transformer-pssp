d_modelArray = [32, 64, 128, 256, 512]
n_layersArray = [1, 2, 3, 4, 5, 6]
dropoutArray = [0.01, 0.1, 0.6]
attentionHeadsArray = [8, 16]

d_modelArray.reverse()
n_layersArray.reverse()
dropoutArray.reverse()
attentionHeadsArray.reverse()


i = 0
print("NN,d_model,n_layers,dropout,n_head")
for d_model in d_modelArray:
        for n_layers in n_layersArray:
            for dropout in dropoutArray:
                print(i, d_model, n_layers, dropout, 4, sep=",")
                i = i + 1