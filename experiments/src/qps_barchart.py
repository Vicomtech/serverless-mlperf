import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path
from utils import benchmark_utils as bu
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(height, rect.get_x() + rect.get_width() / 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
def show_barchart_qps_vs_memory_both_benchpyte_horizontal(mobi_list,
                                                          ssd_list,
                                                          mobi_color,
                                                          ssd_color,
                                                          title):

    memory_labels = ["768", "1536", "2240", "3008"]
    fig, ax2, = plt.subplots(dpi=300)
    x = np.arange(len(memory_labels))
    width = 0.12
    rects3 = ax2.barh(x -2*width, mobi_list["openvino"], width,
                      color=(mobi_color["ov"][0],
                             mobi_color["ov"][1],
                             mobi_color["ov"][2],
                             mobi_color["ov"][3]), label='MobileNetV1 IE-IR',edgecolor="black")
    rects4 = ax2.barh(x - width, mobi_list["tf"], width, color=(mobi_color["tf"][0],
                                                                  mobi_color["tf"][1],
                                                                  mobi_color["tf"][2],
                                                                  mobi_color["ov"][3]),
                       label='MobileNetV1 OCV-TF', edgecolor="black")
    rects5 = ax2.barh(x, mobi_list["caffe"], width, color=(mobi_color["caffe"][0],
                                                                    mobi_color["caffe"][1],
                                                                    mobi_color["caffe"][2],
                                                                    mobi_color["caffe"][3]),
                                                                    label='MobileNetV1 OCV-CF',edgecolor="black")
    rects6 = ax2.barh(x + width, ssd_list["openvino"], width,
                     color=(mobi_color["ov"][0] , ssd_color["ov"][1], ssd_color["ov"][2] , ssd_color["ov"][3]),
                                                                label='SSDMobileNetV1 IE-IR',edgecolor="black")
    rects7 = ax2.barh(x + 2*width, ssd_list["tf"], width, color=(ssd_color["tf"][0],
                                                                 ssd_color["tf"][1],
                                                                 ssd_color["tf"][2],
                                                                 ssd_color["tf"][3]),
                     label='SSDMobileNetV1 OCV-TF', edgecolor="black")
    rects8 = ax2.barh(x+3*width, ssd_list["caffe"], width, color=(ssd_color["caffe"][0],
                                                                  ssd_color["caffe"][1],
                                                                  ssd_color["caffe"][2] ,
                                                                  ssd_color["caffe"][3]),
                     label='SSDMobileNetV1 OCV-CF', edgecolor="black")

    autolabel(rects3, ax2)
    autolabel(rects4, ax2)
    autolabel(rects5, ax2)
    autolabel(rects8, ax2)
    autolabel(rects6, ax2)
    autolabel(rects7, ax2)
    ax2.set_ylabel('Memory(MB)')
    ax2.set_title(title)
    ax2.set_yticks(x)
    ax2.set_yticklabels(memory_labels)
    ax2.invert_yaxis()
    ax2.set_xlim([30,100])
    ax2.legend(loc = 'lower right', prop={'size': 10})
    ax2.set_xlabel('QPS(Queries per second)')
    barchart_file = os.path.join(os.getcwd(), "fig3-elordi.pdf")
    plt.savefig(barchart_file)

data = bu.prepare_data_for_latencies_barchart(os.path.join(os.getcwd(), "results.csv"),
                                              prefix="qps")
# show_barchart_qps_vs_memory(data, "mobilenet", "MobilenetV1 latency")
# show_barchart_qps_vs_memory(data, "ssd", "SSDMobilenetV1 latency")
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 13}

matplotlib.rc('font', **font)
ssd_qps_colors = {
        "ov":[239/255, 238/255, 157/255, 0.9],
        "tf":[209/255, 234/255, 163/255, 0.9],
        "caffe":[219/255, 198/255, 235/255, 0.9]
    }
mobi_qps_colors = {
        "ov":[168/255, 230/255, 207/255, 0.9],
        "tf":[255/255, 211/255, 182/255, 0.9],
        "caffe":[234/255, 144/255, 122/255, 0.9]
    }
show_barchart_qps_vs_memory_both_benchpyte_horizontal(data["mobilenet"],
                                                      data["ssd"],
                                                      mobi_qps_colors,
                                                      ssd_qps_colors,
                                                      "Throughput benchmark: MobileNetV1 & SSDMobileNetV1")
plt.show()