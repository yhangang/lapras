# coding:utf-8

import pandas as pd
import numpy as np
import math
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from .utils.func import to_ndarray
from .stats import probability

from lapras.utils import count_point


def get_params_data_dict(model_name):
    with open(model_name, 'r') as f:
        params_data = f.read()
    f.close()

    params_data_valid_s = re.findall('(python_out.*?)\nAverage', params_data, re.S)
    if params_data_valid_s:
        params_data_valid = params_data_valid_s[-1]
    else:
        print ('model文件有误，请检查文件格式')
        return
    params_data_list = params_data_valid.split('\n')

    params_data_dict = dict()
    for params_data_one in params_data_list:
        if not params_data_one:
            continue
        if 'python_out' not in params_data_one and 'cnt_bad_rate' not in params_data_one:
            continue
        if 'python_out[' in params_data_one:
            continue
        params_data_one = params_data_one.replace('python_out', '').replace(' ', '').replace('"', '')
        param_data_one_list = params_data_one.split('\t')
        param_name = param_data_one_list[0].split(':')[0]
        param_bond = param_data_one_list[0].split(':')[1]
        param_good = param_data_one_list[1].split(':')[1]
        param_bad = param_data_one_list[2].split(':')[1]
        param_bad_rate = param_data_one_list[3].split(':')[1]
        param_IV = param_data_one_list[4].split(':')[1]
        params_data_dict[param_name] = dict()
        params_data_dict[param_name]['bond'] = eval(param_bond)
        params_data_dict[param_name]['good'] = eval(param_good)
        params_data_dict[param_name]['bad'] = eval(param_bad)
        params_data_dict[param_name]['bad_rate'] = eval(param_bad_rate)
        params_data_dict[param_name]['IV'] = eval(param_IV)
    return params_data_dict


def show_singe_param_pic(params_data_dict, param):
    params_data = params_data_dict.get(param)
    if params_data != None:
        y_count = [int(g_i) + int(params_data['bad'][index_g_i]) for index_g_i, g_i in enumerate(params_data['good'])]
        y_rate = [float(br_i) for br_i in params_data['bad_rate']]
        x = list(range(len(y_count)))
        ticks = ['[' + params_data['bond'][index_bd] + ',' + bd + ']' for index_bd, bd in enumerate(params_data['bond'][1:])]

        plt_show(x, ticks,y_count, y_rate, param)


def show_single(model_name, ParamsShow):
    if isinstance(ParamsShow, str):
        ParamsShow = [ParamsShow]

    if not isinstance(ParamsShow, list):
        return
    params_data_dict = get_params_data_dict(model_name)
    for param in ParamsShow:
        # if len(params_data_dict[param]['bond']) < 6:
        #     continue
        print (param)
        show_singe_param_pic(params_data_dict, param)


def show_mutil(model_name):
    params_data_dict = get_params_data_dict(model_name)
    params_data_key_valid = list(params_data_dict.keys())
    # params_data_key_valid = list(params_data_dict.keys())[:800]

    i = 0
    for param in params_data_key_valid:
        # if len(params_data_dict[param]['bond']) < 6:
        #     continue
        print (param)
        show_singe_param_pic(params_data_dict, param)
        i += 1

def bin_plot(frame, col=None, target='target', **kwargs):
    """plot for bins

    Args:
        frame (DataFrame)
        x (str): column in frame that will be used as x axis
        target (str): target column in frame

    """
    frame = frame.copy()
    group = frame.groupby(col)
    table = group[target].agg(['sum', 'count']).reset_index()
    table.columns = [col,'bad_count','count']
    table['bad_rate'] = table['bad_count'] / table['count']

    plt_show(table.index, table[col], table['count'], table['bad_rate'], title=col, **kwargs)


def score_plot(frame, score='score', target='target',score_bond=None, **kwargs):
    """plot for scores

    Args:
        frame (DataFrame)
        x (str): column in frame that will be used as x axis
        target (str): target column in frame

    """
    if score_bond is None:
        max_value = int(frame[score].max())+1
        min_value = int(frame[score].min())
        score_bond = np.arange(min_value,max_value,step=30)
        score_bond[-1] = max_value

    # 计算 区间数量 区间坏账率
    x, ticks, y_count, y_rate = count_point(frame, score_bond, score, target, **kwargs)

    # 画图显示 区间数量 区间坏账率
    plt_show(x, ticks, y_count, y_rate, **kwargs)


def plt_show(x, ticks, y_count, y_rate, title="Score Distribute And Bad Rate", x_label="Score Bonds",
             y_label_left="Sample Counts", y_label_right="Bad Rates", fontsize=15, output=False):
    '''
    画 柱状图 和 折线图
    :param x: 区间分段 1,2,3,4
    :param ticks: 区间名称['[300, 400)', '[400, 500)',  '[500, 1000)']
    :param y_count: 区间 数量， 表示评分在此区间内的样本数量
    :param y_rate: 区间 坏账率
    :param graph_title: 图表标题
    :param x_label: 横坐标标题
    :param y_label_left: 左边从坐标标题
    :param y_label_right: 右边从坐标标题
    :param fontsize: 字体大小
    '''
    # 设置字体、图形样式
    # sns.set_style("whitegrid")
    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # matplotlib.rcParams['font.family'] = 'sans-serif'
    # matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.fontsize = fontsize



    # 是否显示折线图
    line_flag = True

    y1 = y_count
    y2 = y_rate
    # 设置图形大小
    plt.rcParams['figure.figsize'] = (18.0, 9.0)

    fig = plt.figure()

    # 画柱子
    ax1 = fig.add_subplot(111)
    # alpha透明度， edgecolor边框颜色，color柱子颜色 linewidth width 配合去掉柱子间距
    ax1.bar(x, y1, alpha=0.8, edgecolor='k', color='#3399FF',linewidth=1, width =1)
    # 获取 y 最大值 最高位 + 1 的数值 比如 201取300，320取400，1800取2000
    y1_lim = int(str(int(str(max(y1))[0]) + 1) + '0' * (len(str(max(y1))) - 1))

    # 设置 y轴 边界
    ax1.set_ylim([0, y1_lim])
    # 设置 y轴 标题
    ax1.set_ylabel(y_label_left, fontsize='15')
    ax1.set_xlabel(x_label,fontsize='15')
    # 将分值标注在图形上
    for x_i, y_i in  zip(x, y1):
        ax1.text(x_i, y_i + y1_lim/20, str(y_i), ha='center', va='top', fontsize=13, rotation=0)

    # 设置标题
    ax1.set_title(title, fontsize='20')
    plt.yticks(fontsize=15)
    # plt.xticks(x, y)
    plt.xticks(fontsize=12)

    # 画折线图
    if line_flag:
        ax2 = ax1.twinx()  # 这个很重要噢
        ax2.plot(x, y2, 'r', marker='*', ms=0)

        # ax2.set_xlim([-0.5, 3.5])
        try:
            y2_lim = (int(max(y2) * 10) + 1) / 10
        except:
            y2_lim = 1
        ax2.set_ylim([0, y2_lim])
        ax2.set_ylabel(y_label_right, fontsize='15')
        ax2.set_xlabel(x_label,fontsize='15')
        for x_i, y_i in  zip(x, y2):
            ax2.text(x_i, y_i+y2_lim/20 , '%.2f%%'%(y_i *100), ha='center', va='top', fontsize=13, rotation=0)
    plt.yticks(fontsize=15)
    plt.xticks(x, ticks)
    plt.xticks(fontsize=15)

    # 是否显示网格
    plt.grid(False)

    # 保存图片 dpi为图像分辨率
    # plt.savefig('分数分布及区间坏账率.png', dpi=600, bbox_inches='tight')
    # 显示图片
    plt.show()


def radar_plot(data=[], title='Radar Graph', radar_labels=['dimension1','dimension2','dimension3','dimension4', 'dimension5'],
               figsize=(6,6), fontsize=15):
    """

    Args:
        data: List of data
        title: the name of the graph
        radar_labels: the name of each dimension
        figsize: figsize
        fontsize: fontsize
    Returns:

    """
    angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False)
    fig = plt.figure(figsize=figsize,facecolor = "white")
    plt.subplot(111, polar = True)
    plt.ylim(0,1)
    plt.plot(angles, data,'o-',linewidth=1, alpha=0.2)
    plt.fill(angles, data, alpha=0.25)
    plt.thetagrids(angles*180/np.pi, radar_labels,fontsize=fontsize)
    plt.figtext(0.52, 0.95,title , ha='center', size=25)
    # plt.setp(legend.get_texts(), fontsize='large')
    plt.grid(True)
    # plt.savefig('holland_radar.jpg')
    plt.show()

