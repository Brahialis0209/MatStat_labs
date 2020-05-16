import numpy as np
import matplotlib.pyplot as plt
import coutse_work.ripper
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
pyglobus_dir = os.path.join(current_dir, "pyglobus", "python")
sht_dir = os.path.join(current_dir, "sht")
sys.path.append(pyglobus_dir)
try:
    import pyglobus
except ImportError as e:
    print("Cannot import pyglobus from %s, exiting" % pyglobus_dir)
    sys.exit(1)

# боксплот для нахождения выбросов
def whisker_bounds(signal):
    upper_quartile = np.percentile(signal, 75)
    lower_quartile = np.percentile(signal, 25)

    iqr = upper_quartile - lower_quartile
    upper_bound = upper_quartile + 1.5 * iqr
    lower_bound = lower_quartile - 1.5 * iqr
    return lower_bound, upper_bound


# скользяцее среднее, с помощью которого извлекаем тренд
def moving_average(signal, width):
    new_signal = np.zeros_like(signal)
    for i in range(0, signal.shape[0]):
        new_signal[i] = np.mean(signal[max(0, i - width): min(signal.shape[0], i + width)])
    return new_signal


# является ли значение в сигнале выбросом, покомпонентную разницу между сигналом и трендом загоняем в боксплот
def is_outlier(signal, width):
    # берем разницу между сигналом и трендом в окне
    diff = signal - moving_average(signal, width)
    lower_bound, upper_bound = whisker_bounds(diff)

    return (diff <= lower_bound) | (diff >= upper_bound)


# заменяет найденные выбросы медианным значением
def replace_outliers(signal, width):
    sgn = signal.copy()
    is_out = is_outlier(sgn, width)
    for idx in np.where(is_out)[0]:
        l = max(0, idx - width)
        r = min(sgn.shape[0], idx + width)
        ss = np.where(~is_out[l:r], sgn[l:r], np.nan)
        sgn[idx] = np.nanmedian(ss)
    return sgn


# запускает два раза с шириной окна сначала 10%, потом 2%
# с помощью 10% убираем большие выбросы
# с помощью 2% убираем локальные выбросы, которые появились уже после уборки больших
def confident_outliers_replace(signal):
    # choose width with 10% of data
    replaced = replace_outliers(signal, len(signal) // 10)
    # choose width with 2% of data
    replaced = replace_outliers(replaced, len(signal) // 50)
    return replaced


# идем слева направо, потом справа налево с каким-то окном размера step
def get_useful_signal_boundaries(signal, bin_edges, step, threshold):
    for i in range(0, len(signal), step):
        # берем часть сигнала размера step
        part = signal[i:i + step]
        # смотрим какие значениях сигнала находятся внутри границ
        idxs = (bin_edges[0] <= part) & (part <= bin_edges[1])
        if idxs.sum() / step < threshold:
            left = i
            break

    for i in range(len(signal), step, -step):
        part = signal[i - step:i]
        idxs = (bin_edges[0] <= part) & (part <= bin_edges[1])
        if idxs.sum() / step < threshold:
            right = i
            break

    return left, right


def process(timestamp, signal, step=100, threshold=0.85):
    signal = signal.copy()
    # 1 этап получение гистограммы всего сигнала
    # edges хранит в себе границы
    hist, edges = np.histogram(signal)
    # 2 наиболее высоких столбика
    # idx1 - самый большой
    # idx2 - следующий после большого
    idx2, idx1 = np.argsort(hist)[-2:]
    # находит границы полезного сигнала, без шумов
    left, right = get_useful_signal_boundaries(signal, (edges[idx1], edges[idx1 + 1]),
                                               step=step, threshold=threshold)
    # если два столбика отличаются высотой не больше, чем на 20 процентов
    if abs(hist[idx1] - hist[idx2]) / signal.shape[0] < 0.2:
        l1, r1 = get_useful_signal_boundaries(signal, (edges[idx2], edges[idx2 + 1]),
                                              step=step, threshold=threshold)
        # Границы полезного сигнала
        left = max(left, l1)
        right = min(right, r1)

    cutted_timestamp, cutted_signal = timestamp[left:right], signal[left:right].copy()
    # функция замены выбросов
    norm_signal = confident_outliers_replace(cutted_signal)
    # plt.clf()
    # plt.plot(timestamp, signal, label='Оригинал')
    # plt.plot(timestamp[:left], signal[:left], c='r')
    # plt.plot(timestamp[right:], signal[right:], c='r', label='Шум')
    #
    # to_plot_idx = np.nonzero(cutted_signal-norm_signal)[0]
    # plt.scatter(cutted_timestamp[to_plot_idx], cutted_signal[to_plot_idx], s=20, c='g', label = 'Выбросы')
    # plt.legend()
    # plt.show()
    return cutted_timestamp, norm_signal


def research_sustainability(x, amplitude, dir_name,  SHT_NUMBER, signal_n):
    iter_num = 0
    for ths in range(1, 100, 10):

        dev_ths = ths / 100
        plt.subplot(2, 2, 1)
        plt.title(f'Signal {signal_n}, threshold =  {dev_ths}:')
        plt.plot(x, amplitude)

        plt.subplot(2, 2, 2)
        plt.hist(amplitude)

        xx, sgn = process(x, amplitude, threshold=dev_ths)
        plt.subplot(2, 2, 3)
        plt.title(f'Processed signal {signal_n}')
        plt.plot(xx, sgn)

        plt.subplot(2, 2, 4)
        plt.hist(sgn)

        plt.tight_layout()
        Title_str = current_dir + '/res/' + dir_name + 'threshold = ' + str(dev_ths) + ' Num = ' + str(iter_num) + ' ShotNo= ' + str(SHT_NUMBER) + ' Signal= ' + str(signal_n) + '.png'
        plt.savefig(Title_str)
        plt.clf()
        iter_num += 1

    for ths in range(97, 100, 1):
        dev_ths = ths / 100
        plt.subplot(2, 2, 1)
        plt.title(f'Signal {signal_n}:')
        plt.plot(x, amplitude)

        plt.subplot(2, 2, 2)
        plt.hist(amplitude)

        xx, sgn = process(x, amplitude, threshold=dev_ths)
        plt.subplot(2, 2, 3)
        plt.title(f'Processed signal {signal_n}')
        plt.plot(xx, sgn)

        plt.subplot(2, 2, 4)
        plt.hist(sgn)

        plt.tight_layout()
        Title_str = current_dir + '/res/' + dir_name + 'threshold = ' + str(dev_ths) + ' Num = ' + str(
            iter_num) + ' ShotNo= ' + str(SHT_NUMBER) + ' Signal= ' + str(signal_n) + '.png'
        plt.savefig(Title_str)
        plt.clf()
        iter_num += 1



# --------------------------------------
# dir_name = 'sht38515/'
# date_path = pyglobus_dir = os.path.join(current_dir, 'date', 'sht38515.sht')
# signal_n = 20
# SHT_NUMBER = 38515
# sht_reader = pyglobus.util.ShtReader(date_path)
# signal = sht_reader.get_signal(signal_n)
# # data_all = ripper.extract('', 38515)[0]
#
# x, amplitude = signal.get_data_x(), signal.get_data_y()
# x, amplitude = np.array(x), np.array(amplitude)
#
# research_sustainability(x, amplitude, dir_name,   SHT_NUMBER, signal_n)

# plt.subplot(2, 2, 1)
# plt.title(f'Signal {signal_n}:')
# plt.plot(x, amplitude)
#
# plt.subplot(2, 2, 2)
# plt.hist(amplitude)
#
# xx, sgn = process(x, amplitude)
# plt.subplot(2, 2, 3)
# plt.title(f'Processed signal {signal_n}')
# plt.plot(xx, sgn)
#
# plt.subplot(2, 2, 4)
# plt.hist(sgn)
#
# plt.tight_layout()
# SHT_NUMBER = 38515
# Title_str = 'ShotNo= ' + str(SHT_NUMBER) + ' Signal= ' + str(signal_n) + '.png'
# plt.savefig(Title_str)
# plt.clf()
# # # --------------------------------------
# #
# #
# #
# #
# #
# # # --------------------------------------

dir_name = 'sht38852/'
date_path = pyglobus_dir = os.path.join(current_dir, 'date', 'sht38852.sht')
signal_n = 24
SHT_NUMBER = 38852
# data_all = ripper.extract('', 38852)[0]
sht_reader = pyglobus.util.ShtReader(date_path)
signal = sht_reader.get_signal(signal_n)

x, amplitude = signal.get_data_x(), signal.get_data_y()
x, amplitude = np.array(x), np.array(amplitude)
research_sustainability(x, amplitude, dir_name, SHT_NUMBER, signal_n)

#
# plt.subplot(2, 2, 1)
# plt.title(f'Signal {signal_n}:')
# plt.plot(x, amplitude)
#
# plt.subplot(2, 2, 2)
# plt.hist(amplitude)
#
# xx, sgn = process(x, amplitude)
# plt.subplot(2, 2, 3)
# plt.title(f'Processed signal {signal_n}')
# plt.plot(xx, sgn)
#
# plt.subplot(2, 2, 4)
# plt.hist(sgn)
#
# plt.tight_layout()
# SHT_NUMBER = 38852
# Title_str = 'ShotNo= ' + str(SHT_NUMBER) + ' Signal= ' + str(signal_n) + '.png'
# plt.savefig(Title_str)
# --------------------------------------
