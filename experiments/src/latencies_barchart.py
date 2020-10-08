import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path
from os import path
from utils import benchmark_utils as bu
def autolabel(rects, ax, offset=0):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2+offset, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
def show_barchart_latencies_vs_memory(ax2, cvs_list, prefix, title):
    memory_labels = ["768", "1536", "2240", "3008"]

    x = np.arange(len(memory_labels))
    width = 0.25
    rects3 = ax2.bar(x - width, cvs_list[prefix]["openvino"], width, color=(168/255, 230/255, 207/255, 0.9), label='IE-IR')
    rects5 = ax2.bar(x, cvs_list[prefix]["caffe"], width, color=(220/255, 237/255, 193/255, 0.9), label='OCV-CF')
    rects4 = ax2.bar(x + width, cvs_list[prefix]["tf"], width, color=(255 / 255, 211 / 255, 182 / 255, 0.9),
                     label='OCV-TF')
    autolabel(rects3, ax2)
    autolabel(rects4, ax2, 0.15)
    autolabel(rects5, ax2)
    ax2.set_ylabel('Latency time (msecs)')
    ax2.set_title(title)
    ax2.set_xticks(x)
    ax2.set_ylim([0,550])
    ax2.set_xticklabels(memory_labels)
    ax2.legend()
    ax2.set_xlabel('Memory (MB)')
    barchart_file = os.path.join(os.getcwd(), prefix + "barchart.png")
    plt.savefig(barchart_file)
    return ax2
     #plt.show()
def show_benchmark_latency_figure(filename, data, prefix, title):
    fig, ax1 = plt.subplots(dpi=300)
    ax1 = show_barchart_latencies_vs_memory(ax1, data, prefix, title)
    plt.tight_layout()
    absolute_filename= os.path.join(os.getcwd(), filename)
    plt.savefig(absolute_filename)
def show_unified_figure(data, prefixes = [], titles = []):
    fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(10, 4), dpi=300)
    #for prefix, title in zip(prefixes, titles):
    ax1 = show_barchart_latencies_vs_memory(ax1, data, prefixes[0], titles[0])
    ax2 = show_barchart_latencies_vs_memory(ax2, data, prefixes[1], titles[1])
    barchart_file_svg = os.path.join(os.getcwd(), "fig1-elordi.svg")
    barchart_file_png = os.path.join(os.getcwd(),  "fig1-elordi.png")
    barchart_file_pgf = os.path.join(os.getcwd(), "experiments",  "fig1-elordi.pdf")
    plt.tight_layout()
    plt.savefig(barchart_file_svg)
    plt.savefig(barchart_file_pgf)
    plt.show()
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 13}

matplotlib.rc('font', **font)
data = bu.prepare_data_for_latencies_barchart(os.path.join(os.getcwd(),"results.csv"), prefix="latency")
titles = ["Latency Benchmark: MobilenetV1",
          "Latency Benchmark: SSDMobilenetV1"]
prefixes = ["mobilenet", "ssd"]
#filenames = ["fig2-1-elordi.png","fig2-2-elordi.png" ]
filenames = ["fig2-1-elordi.pdf","fig2-2-elordi.pdf"]
#filenames = ["fig2-1-elordi.pdf","fig2-2-elordi.pdf"]
for title, prefix,  filename in zip(titles, prefixes,  filenames):
    show_benchmark_latency_figure(filename, data, prefix, title)
#show_unified_figure(data, prefixes, titles)
plt.show()