import numpy as np
import visdom
import os
import plotly.express as px
# import indicator

vis = visdom.Visdom(port=5501)


def FindContainList(strList, conList):
    out = []
    for strs in strList:
        if conList in strs:
            out.append(strs)

    return out


def PlotWithPlotly(path=None):
    if path is None:
        listname = os.listdir('log/')
        listname.remove('.DS_Store')
        listname.remove('baseline')
        listname.sort()
        print(listname)
    else:
        listname = [path]

    for path in listname:

        listname = os.listdir(path)
        listContaininput = FindContainList(listname, 'input')[-1]
        listContainlabel = FindContainList(listname, 'label')[-1]
        listContainlatent = FindContainList(listname, 'latent')[-1]

        inputdatapath = path+'/'+listContaininput
        labeldatapath = path+'/'+listContainlabel
        latentdatapath = path+'/'+listContainlatent

        data_input = np.loadtxt(inputdatapath)
        data_latent = np.loadtxt(latentdatapath)
        data_label = np.loadtxt(labeldatapath)

        fig = px.scatter_3d(
            x=data_latent[:, 0],
            y=data_latent[:, 1],
            z=data_latent[:, 2],
            color=data_label.astype(np.int32),
            size=np.ones_like(data_label)*0.4,
            opacity=1
        )
        fig.write_html(path+'/'+listContainlatent+'avis.html')


        # indi = indicator.GetIndicator(data_input, data_latent)
        # print(indi)
if __name__ == "__main__":
    PlotWithPlotly()
