import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from mat4py import loadmat
from pyriemann.estimation import Covariances
from scipy.signal import correlate
import statsmodels.api as sm

# Бинаризация .mat файлов
def get_bin_files():
    for i in tqdm(range(52)):
        if len(str(i+1)) == 1:
            sub_number = 's0' + str(i+1)
        else:
            sub_number = 's' + str(i+1)
            
        if not os.path.exists('dataset_hand_MI_bin/' + sub_number + '.pkl'):
            path = os.path.join(os.path.dirname(os.getcwd()), 'code', 'dataset_left-right_hand_MI', sub_number+'.mat')
            data_si = loadmat(path)
            save_object(data_si, 'dataset_hand_MI_bin/' + sub_number + '.pkl')


# Получение даты в .npy
def get_data():
        
    for i in tqdm(range(52)):
        data = []
        data_loc = []
        if len(str(i+1)) == 1:
                sub_number = 's0' + str(i+1) + '.pkl'
        else:
            sub_number = 's' + str(i+1) + '.pkl'
            
            
        file_name = 'dataset_hand_MI_bin/' + sub_number
           
        data_si = load_object(file_name)
        loc_si = data_si['eeg']['senloc']
        loc_si = np.array([np.array(loc_si[k]) for k in range(len(loc_si))])
            

        ts_si_0 = data_si['eeg']['imagery_left']
        ts_si_1 = data_si['eeg']['imagery_right']
        ts_si_0 = np.array([np.array(ts_si_0[k]) for k in range(len(ts_si_0))])
        ts_si_1 = np.array([np.array(ts_si_1[k]) for k in range(len(ts_si_1))])

        # Оставляем только сигналы EEG
        ts_si_0 = ts_si_0[:64]
        ts_si_1 = ts_si_1[:64]

        # (Временной ряд, метка класса, номер объекта)
        data.append((ts_si_0, 0, i))
        data.append((ts_si_1, 1, i))

    data_loc = np.array(data_loc)
    data = np.array(data)
    np.save('dataset_hand_MI_bin/data_hand_MI.npy', data)
    np.save('dataset_hand_MI_bin/data_hand_MI_location.npy', data_loc)


def save_object(obj, filename):
    # Overwrites any existing file.
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
        return obj

class dataset():
    def __init__(self):
        pass

    def get_sub_data(self, file_name):
        data = load_object(file_name)
        # Получение временных рядов для каждого класса
        ts_0 = np.array(data['eeg']['imagery_left'])
        ts_1 = np.array(data['eeg']['imagery_right'])
        # Оставляем только сигналы EEG
        ts_0 = ts_0[:64]
        ts_1 = ts_1[:64]
        return ts_0, ts_1 
    
    # Получаем корреляционные матрицы и метки классов после аугментации временных рядов
    def get_augmented_Covariances(self, window_size = 3584):
        X_Cov = []
        y = []
        n_sub = []
        for i in tqdm(range(52)):
            sub_number = 's0'+str(i+1)+'.pkl' if len(str(i+1))==1 else 's'+str(i+1)+'.pkl'
            file_name = 'dataset_hand_MI_bin/' + sub_number
            ts_0, ts_1 = self.get_sub_data(file_name)
            ts_0 = np.array([ts_0[:, i:i+window_size] for i in range(0, ts_0.shape[1], window_size)])
            ts_1 = np.array([ts_1[:, i:i+window_size] for i in range(0, ts_1.shape[1], window_size)])
            for part_ts_0, part_ts_1 in zip(ts_0, ts_1):
                X_Cov.append(Covariances().fit_transform(np.expand_dims(part_ts_0, axis=0))[0])
                X_Cov.append(Covariances().fit_transform(np.expand_dims(part_ts_1, axis=0))[0])
                y.append(0)
                y.append(1)
                n_sub.append(i)
                n_sub.append(i)
        return np.array(X_Cov), np.array(y), np.array(n_sub)



# eeg_data_class: 'imagery_left', 'imagery_right'
# sub_path: 'dataset_hand_MI_bin/s##.pkl'
class correlation_analysis():
    def __init__(self, sub_path: str, eeg_data_class: str):
        self.data = load_object(sub_path)
        # Координаты расположения датчиков 
        self.loc = np.array(self.data['eeg']['senloc'])
        # Данные ЭЭГ испытуемого
        self.ts = np.array(self.data['eeg'][eeg_data_class])[:64]
        self.eeg_data_class = eeg_data_class

    def plot_signals_and_corrs(self, main_sensor_num: int, additional_sensor_num: int, centered=True):
        # Номера датчиков выбираются из [0, ..., 63]
        if centered:
            ts = self.ts - self.ts.mean(axis=1, keepdims=True)
        else:
            ts = self.ts
        # найдем ближайший и самый удаленный датчик от main_sensor_num
        dists = np.array([np.linalg.norm(self.loc[main_sensor_num] - cord) for cord in self.loc])
        masked_dists = np.ma.masked_equal(dists, 0)  # Создаем маску, заменяя все нулевые элементы на значение fill_value
        nearest_num = np.argmin(masked_dists)
        distant_num = np.argmax(dists)
        print(f'Номер ближайшего датчика: {nearest_num + 1}, расстояние: {round(dists[nearest_num], 3)}')
        print(f'Номер самого удаленного датчика: {distant_num + 1}, расстояние: {round(dists[distant_num], 3)}')

        imagery_event = np.array(self.data['eeg']['imagery_event'])
        # Индексы моментов воображения действия объектом
        imagery_event_indexes = np.array([i for i in range(len(imagery_event)) if imagery_event[i] == 1])

        ts1 = ts[main_sensor_num]
        ts2 = ts[nearest_num]
        ts3 = ts[distant_num]
        ts4 = ts[additional_sensor_num]
        
        plt.rcParams['figure.figsize'] = (12, 8)
        fig, (ax_ts1, ax_ts2, ax_ts3, ax_ts4) = plt.subplots(4, 1, sharex=True, layout='constrained')
        ax_ts1.plot(ts1, label = 'EEG signal')
        ax_ts1.plot(imagery_event_indexes, ts1[imagery_event_indexes], 'ro', label = 'imagery event')
        ax_ts1.set_title('EEG signal from sensor '+ str(main_sensor_num + 1), fontsize = '16')
        ax_ts1.legend(fontsize = '14')
        ax_ts2.plot(ts2, label = 'EEG signal')
        ax_ts2.plot(imagery_event_indexes, ts2[imagery_event_indexes], 'ro', label = 'imagery event')
        ax_ts2.set_title('EEG signal from nearest sensor '+ str(nearest_num + 1), fontsize = '16')
        ax_ts2.legend(fontsize = '14')
        ax_ts3.plot(ts3, label = 'EEG signal')
        ax_ts3.plot(imagery_event_indexes, ts3[imagery_event_indexes], 'ro', label = 'imagery event')
        ax_ts3.set_title('EEG signal from distant sensor ' + str(distant_num + 1), fontsize = '16')
        ax_ts3.legend(fontsize = '14')
        ax_ts4.plot(ts4, label = 'EEG signal')
        ax_ts4.plot(imagery_event_indexes, ts4[imagery_event_indexes], 'ro', label = 'imagery event')
        ax_ts4.set_title('EEG signal from additional sensor ' + str(additional_sensor_num + 1), fontsize = '16')
        ax_ts4.legend(fontsize = '14')
        ax_ts1.margins(0, 0.1)
        fig.tight_layout()
        plt.show()


        corr12 = sm.tsa.stattools.ccf(ts2, ts1, adjusted=False)
        corr13 = sm.tsa.stattools.ccf(ts3, ts1, adjusted=False)
        corr14 = sm.tsa.stattools.ccf(ts4, ts1, adjusted=False)

        # Remove padding and reverse the order
        corr12 = corr12[0:(len(ts2)+1)][::-1] 
        corr13 = corr13[0:(len(ts3)+1)][::-1]
        corr14 = corr14[0:(len(ts4)+1)][::-1]

        fig, (ax_corr12, ax_corr13, ax_corr14) = plt.subplots(3, 1, sharex=True, layout='constrained')
        ax_corr12.plot(corr12)
        ax_corr12.set_title('Cross-correlation between the ' + str(main_sensor_num+1) + ' and ' + str(nearest_num+1) + ' sensor', fontsize = '16')
        ax_corr13.plot(corr13)
        ax_corr13.set_title('Cross-correlation between the ' + str(main_sensor_num+1) + ' and ' + str(distant_num+1) + ' sensor', fontsize = '16')
        ax_corr14.plot(corr14)
        ax_corr14.set_title('Cross-correlation between the ' + str(main_sensor_num+1) + ' and ' + str(additional_sensor_num+1) + ' sensor', fontsize = '16')
        ax_corr12.margins(0, 0.1)
        ax_corr13.margins(0, 0.1)
        fig.tight_layout()
        plt.show()