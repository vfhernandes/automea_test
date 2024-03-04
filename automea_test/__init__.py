import pandas as pd
import os
from contextlib import redirect_stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with redirect_stderr(open(os.devnull, "w")):
    import tensorflow as tf
import h5py
import numpy as np 
import matplotlib.pyplot as plt
#from analyseMEA import *
from automea import util

class Analysis:


    def __init__(self):

        self.util = util

        # initialize attributes
        self.dataset = []

        self.dataset_filename = ''

        self.dataset_index = None
        
        self.path_to_dataset = ''
        
        self.path_to_csv = ''

        self.path_to_model = ''

        self.output_name = ''

        self.output_folder = None
        
        
        self.wellsIDs = []

        self.wellsLabels = []

        self.well = None

        self.model = None

        self.model_name = None
        
        self.signal = None
        
        self.spikes = None

        self.spikes_binary = None

        self.reverbs = None

        self.reverbs_binary = None

        self.bursts = None

        self.bursts_binary = None
        
        self.samplingFreq = 10_000

        self.total_timesteps_signal = 6_000_000

        self.threshold_params = {'rising'      : 5,
                                'startTime'    : 0, 
                                'baseTime'     : 0.250, #ms 
                                'segments'     : 10}

        self.spikes_params = {'deadtime': 3_000*1e-6}

        self.reverbs_params = {'max_interval_start'   : 15,
                              'max_interval_end'     : 20,
                              'min_interval_between' : 25,
                              'min_duration'         : 20,
                              'min_spikes'           : 5}

        self.bursts_params = {'min_interval_between'   : 300}

        self.plot_name = None


        # define dictionary for well labels to index and vice-versa,
        # based on plate with 4x6 wells
        keys = []
        items = []
        i = 0
        j_labels = ['A', 'B', 'C', 'D']
        k_labels = range(1,7)
        for j in j_labels:
            for k in k_labels:
                keys.append(i)
                items.append(f'{j}{k}')
                i += 1
        self.wellIndexLabelDict = dict(zip(keys, items))
        self.wellLabelIndexDict = dict(zip(items, keys))


        self._pretrained_models = ['signal30.h5']

        

        self.model_params = {'name'           : None,
                             'input_type'     : None,
                             'input_average'  : 30,
                             'window_size'    : 50_000,
                             'window_overlap' : 25_000}

        self.analysis_params = {'save_spikes'      : False,
                                'save_reverbs'     : False,
                                'save_bursts'      : False,
                                'save_net_reverbs' : False,
                                'save_net_bursts'  : False,
                                'save_stats'       : False}


    def files_and_well_csv(self, file):
        filenames_and_wells = pd.read_csv(self.path_to_csv+file, sep = ';')
        self.dataset = []
        for index in filenames_and_wells.index:
            self.dataset.append(filenames_and_wells['filename'][index])
            if filenames_and_wells['wells'][index] == 'all':
                self.wellsLabels = list(self.wellIndexLabelDict.values())
            else:
                self.wellsLabels.append([])
                for well in filenames_and_wells['wells'][index].split(','):
                    self.wellsLabels[index].append(well) 


    
    def loadmodel(self, setting = 'default'):

        if self.model_name is not None:
            
            self.model_params['name'] = self.model_name

            if setting == 'default':
                self.model_params['input_average'] = 30
                self.model_params['window_size'] = 50_000
                self.model_params['window_overlap'] = 25_000

                if 'signal' in self.model_params['name'].lower():
                    self.model_params['input_type'] = 'signal'
                elif 'spikes' in self.model_params['name'].lower():
                    self.model_params['input_type'] = 'spikes'

                if '100' in self.model_params['name'].lower():
                    self.model_params['input_average'] = 100

            elif setting == 'manual': 
                pass

        else:

            if self.model_params['name'] is None: 
            
                self.model_params['name'] = 'signal30.h5'
                self.model_params['input_average'] == 30
                self.model_params['window_size'] = 50_000
                self.model_params['window_overlap'] = 25_000

            
            if self.model_params['name'] in self._pretrained_models:
            
                if 'SIGNAL' in self.model_params['name']:
                    self.model_params['input_type'] = 'signal'
                elif 'SPIKE' in self.model_params['name']:
                    self.model_params['input_type'] = 'spikes'
            
                self.model_params['input_average'] = int(self.model_params['name'][self.model_params['name'].find('_')+1:self.model_params['name'].find('.')])

                self.model_params['window_size'] = 50_000
                self.model_params['window_overlap'] = 25_000

        
        self.model = tf.keras.models.load_model(self.path_to_model+self.model_params['name'], compile = False)



    def loadsignal(self, signal):
        self.signal = signal
        if np.array(signal.ndim) == 1:
            self.total_timesteps_signal = len(signal)
        else:
            self.total_timesteps_signal = np.array(signal).shape[1]

    def loadh5(self, filename = None):
        
        if filename is None:
            self.h5 = h5py.File(self.path_to_dataset + self.dataset, 'r')
        else:
            self.h5 = h5py.File(self.path_to_dataset + filename, 'r')
            self.dataset = filename
            #self.file.append(filename)
        self.infoChannel = np.asarray(self.h5['Data']['Recording_0']['AnalogStream']['Stream_0']['InfoChannel'])
        try:
            self.loadsignal(np.asarray(self.h5['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData'], dtype = np.int32))
            #self.signal = np.asarray(self.h5['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData'], dtype = np.int32)[:,:self.total_timesteps_signal]
        except:
            print('Error when trying to allocate "signal".')
        self.adZero = np.array([self.infoChannel[i][8] for i in range(len(self.infoChannel))]).reshape(len(self.infoChannel), 1)
        self.conversionFactor =  np.array([self.infoChannel[i][10] for i in range(len(self.infoChannel))]).reshape(len(self.infoChannel), 1)
        self.exponent = np.array([self.infoChannel[i][7] for i in range(len(self.infoChannel))]).reshape(len(self.infoChannel), 1)
        self.wellsFromData = np.array([self.infoChannel[i][2] for i in range(len(self.infoChannel))]).reshape(len(self.infoChannel), 1)
        self.wellsFromData = self.wellsFromData.flatten()


    def loadspikes(self, spikes):
        self.spikes = spikes


    def loadwell(self, file:str, well, method = 'default', spikes = False, reverbs = False, bursts = False, net_reverbs = False, net_bursts = False):

        
        if type(well) is str:
            well_id = self.wellLabelIndexDict[well]
        elif type(well) is int:
            well_id = well
        else:            
            print('Well type not valid.')
            return

        self.loadh5(file)
        if well_id not in self.wellsFromData:
            print('Well not found in the dataset.')
            return 
        
        self.signal = self.signal[self.wellsFromData == well_id]
        self.detect_threshold()
        if spikes:
            self.detect_spikes()
            if method == 'model' and self.model is None:
                pass
            else:
                self.detect_reverbs(method=method)
                self.detect_bursts()
                self.detect_net('reverbs')
                self.detect_net('bursts')
        else:
            if reverbs or bursts or net_reverbs or net_bursts:
                print('Reverbs and Bursts need Spikes to be detected.')
                

    def detect_threshold(self):

        if isinstance(self.signal, np.ndarray) is False:
            print('"signal" has to be a numpy array.')
            return 

        if self.signal.ndim == 1: 
            signal = self.signal.reshape(1,-1)
        else:
            signal = self.signal

        startTimestamp = int(self.threshold_params['startTime'] * self.samplingFreq)
        totalTimestamps = int(self.threshold_params['baseTime'] * self.samplingFreq)
        
        stds = np.zeros((len(signal), self.threshold_params['segments']))
        mean = signal.mean(axis = 1)

        for seg in range(self.threshold_params['segments']):
            signal_ = signal[:,startTimestamp:startTimestamp+totalTimestamps]
            stds[:, seg] = (signal_.T - mean).T.std(axis = 1)
            startTimestamp+=totalTimestamps

        thr = self.threshold_params['rising']*stds.min(axis = 1)
        if len(thr) == 1: self.threshold = thr[0]
        self.threshold = thr

    def convert_timestamps_to_binary(self, input_timestamp, input_type = None, size = None):
        
        if input_type not in ['spikes', 'reverbs', 'bursts']:
            print('Incorrect input type! Needs to be "spikes", "reverbs" or "bursts"')
            return
        
        if size == None: size = self.total_timesteps_signal

        if input_type == 'spikes':
            if self.util._has_list(input_timestamp) is False:
            #if np.array(input_timestamp).ndim == 1:
                _binary = np.zeros(size)
                _binary[input_timestamp] = 1
            else:
                _binary = np.zeros((len(input_timestamp), size))
                for channel, timestamps in enumerate(input_timestamp):
                    _binary[channel][timestamps] = 1

        elif input_type == 'reverbs' or input_type == 'bursts':
            if np.array(input_timestamp, dtype = object).ndim in [1,2]:
                _binary = np.zeros(size)
                for item in input_timestamp:
                    _binary[item[0]:item[1]+1] = 1 
            else:
                _binary = np.zeros((len(input_timestamp), size))
                for channel, timestamps in enumerate(input_timestamp):
                    for item in timestamps:
                        _binary[channel][item[0]:item[1]+1] = 1
        
        return _binary
        


    def convert_binary_to_timestamps(self, input_binary, input_type = None):
        if input_type not in ['spikes', 'reverbs', 'bursts']:
            print('Incorrect input type! Needs to be "spikes", "reverbs", or "bursts"')
            return

        if input_type == 'spikes':
            if np.array(input_binary).ndim == 1:
                return list(np.where(input_binary == 1)[0])
            else:
                timestamps_ = []
                for spikes_bin in input_binary:
                    timestamps_.append(list(np.where(spikes_bin == 1)[0]))
                return timestamps_

        elif input_type == 'reverbs' or input_type == 'bursts':
            if np.array(input_binary).ndim == 1:
                return self._convert_binary_reverb_to_timestamp(input_binary)
            else:
                sequence = []
                for input_channel in input_binary:
                    sequence.append(self._convert_binary_reverb_to_timestamp(input_channel))
                return sequence

    def _convert_binary_reverb_to_timestamp(self, input_binary):
        start_reverb = False
        reverbs_ = []
        for i, item in enumerate(input_binary):
            if item == 1 and not start_reverb:
                reverbs_.append([i])
                start_reverb = True
            elif item == 0 and start_reverb:
                reverbs_[-1].append(i-1)
                start_reverb = False
        return reverbs_




    def _detect_spikes(self, signal, threshold):
        ####################################################################################################################################
        ####################################################################################################################################
        # maybe this could be improved, takes too long for a whole file
        dead = self.spikes_params['deadtime']*self.samplingFreq
        spikes = []
        for i in range(len(signal)):
            if(not len(spikes)):    
                if(abs(signal[i])>threshold):
                    spikes.append(i)
            else:
                if(abs(signal[i])>threshold and (abs(i - spikes[-1]) > dead)):
                    spikes.append(i)    
        
        return spikes



    def detect_spikes(self):

        if isinstance(self.signal, np.ndarray) is False:
            print('"signal" attribute has to be a numpy array.')
            return 
        
        if self.signal.ndim == 1:
            self.spikes = self._detect_spikes(self.signal, self.threshold)
            self.spikes_binary = self.convert_timestamps_to_binary(self.spikes, input_type = 'spikes')
        else:
            self.spikes =  list(map(self._detect_spikes, self.signal, self.threshold))
            self.spikes_binary = np.zeros((len(self.spikes), self.total_timesteps_signal))
            for channel, spikes_ in enumerate(self.spikes):
                self.spikes_binary[channel] = self.convert_timestamps_to_binary(spikes_, input_type = 'spikes')
        


    
    def reverbs_params_default(self):
        self.reverbs_params['max_interval_start'] = 15
        self.reverbs_params['max_interval_end'] = 20
        self.reverbs_params['min_interval_between'] = 25
        self.reverbs_params['min_duration'] = 20
        self.reverbs_params['min_spikes'] = 5


    def reverbs_params_manual(self, params):
        self.reverbs_params['max_interval_start'] = params[0]
        self.reverbs_params['max_interval_end'] = params[1]
        self.reverbs_params['min_interval_between'] = params[2]
        self.reverbs_params['min_duration'] = params[3]
        self.reverbs_params['min_spikes'] = params[4]



    def detect_reverbs(self, method = 'default'): 


        if method == 'default':
            self.reverbs_params_default()


        if self.spikes is None: self.detect_spikes() # detect spikes if "detect_bursts" is called but no spikes are defined

        if method == 'default' or method == 'manual':
            if self.util._has_list(self.spikes) is False:
                self.reverbs = self._detect_reverbs(self.spikes)
                self.reverbs_binary = self.convert_timestamps_to_binary(self.reverbs, input_type = 'reverbs')
            else:
                self.reverbs = list(map(self._detect_reverbs, np.array(self.spikes, dtype = object)))
                self.reverbs_binary = np.zeros((len(self.reverbs), self.total_timesteps_signal))
                for channel, reverbs_ in enumerate(self.reverbs):
                    self.reverbs_binary[channel] = self.convert_timestamps_to_binary(reverbs_, input_type = 'reverbs')
        
        elif method == 'model':
            
            input_type = self.model_params['input_type']
            if input_type == 'signal':
                model_input = self.signal
                run_once = self.signal.ndim == 1
            elif input_type == 'spikes':
                model_input = self.spikes_binary
                run_once = not self.util._has_list(self.spikes)
            else:
                print('Input type or model not defined!')
                return
                

            if run_once:
                self.reverbs = self._predict_reverbs(model_input, self.spikes_binary)
                self.reverbs_binary = self.convert_timestamps_to_binary(self.reverbs, input_type = 'reverbs')
            else:
                self.reverbs = []
                self.reverbs_binary = np.zeros((len(model_input), len(model_input[0])))
                for channel, channel_input in enumerate(model_input):
                    self.reverbs.append(self._predict_reverbs(channel_input, self.spikes_binary[channel]))
                    self.reverbs_binary[channel] = self.convert_timestamps_to_binary(self.reverbs[-1], input_type = 'reverbs')
                


            
    def _detect_reverbs(self, spikes):

        params_multiplier = 10

        # detect bursts with a max interval to start and end
        reverbs = []
        i = 0
        k = 0
        while (i < len(spikes)-2):
            if(abs(spikes[i] - spikes[i+1]) <= params_multiplier*self.reverbs_params['max_interval_start']):
                reverbs.append([spikes[i]])
                j = i+1
                while(abs(spikes[j] - spikes[j+1]) <  params_multiplier*self.reverbs_params['max_interval_end']):
                    reverbs[k].append(spikes[j])
                    j += 1
                    if(j >= len(spikes) - 1): break
                i = j
                k += 1
            else:
                i += 1      

        # check valid reverbarations with minimum duration and minimum number of spikes
        valid_reverbs = []
        
        for reverb in reverbs:
            if len(reverb) > self.reverbs_params['min_spikes'] and \
            abs(reverb[0] - reverb[-1]) >  params_multiplier*self.reverbs_params['min_duration']:
                valid_reverbs.append([reverb[0], reverb[-1]])

        # merge all reverberations that are closer than the minimum interval
        minIntervalBetweenBursts = self.bursts_params['min_interval_between']
        self.bursts_params['min_interval_between'] = self.reverbs_params['min_interval_between']
        merged_reverbs = self._detect_bursts(valid_reverbs)
        self.bursts_params['min_interval_between'] = minIntervalBetweenBursts

        return merged_reverbs



    def detect_bursts(self):

        if self.util._has_list(self.spikes) is False:
            self.bursts = self._detect_bursts(self.reverbs)
        else:
            self.bursts = list(map(self._detect_bursts, self.reverbs))
            self.bursts_binary = np.zeros((len(self.bursts), self.total_timesteps_signal))
            for channel, bursts_ in enumerate(self.bursts):
                self.bursts_binary[channel] = self.convert_timestamps_to_binary(bursts_, input_type = 'bursts')
        

    def _detect_bursts(self, reverbs):

        if len(reverbs) == 1 or not len(reverbs): return reverbs
        params_multiplier = 10 

        r = np.array(reverbs)
        diff = r[1:,0] - r[:-1,1]
        merge = diff < self.bursts_params['min_interval_between']*params_multiplier
        if not merge.any(): return reverbs

        bursts = []
        index = 0
        while index < len(diff):
            if merge[index]:
                bursts.append([r[index,0],r[index+1,1]])
                index += 1
            else:
                bursts.append(list(r[index]))
            index += 1
            
            if index == len(diff): bursts.append(list(r[-1]))
                
        return self._detect_bursts(bursts)

    ################################################################################################################################################
    ################################################################################################################################################
    
    def _predict_reverbs(self, model_input, spikes_binary):

        if self.model_params['input_type'] not in ['signal', 'spikes']:
            print('Input type not correctly defined!')
            return 
    
        if self.total_timesteps_signal%self.model_params['window_overlap'] != 0:
            print("Number of windows and overlap do not match lenght of input channel (signal or spike)!")
            return

        reverbs_binary_pred_long = np.zeros(self.total_timesteps_signal)

        number_of_windows = self.total_timesteps_signal//self.model_params['window_overlap']

        for window_index in range(number_of_windows - 1):
            window_start = window_index*self.model_params['window_overlap']
            window_end = window_start + self.model_params['window_size']
            input_in_window = model_input[window_start:window_end]
            
            spikes_in_window = spikes_binary[window_start:window_end]

            if self.model_params['input_type'] == 'signal':
                input_in_window = self.reduce_norm_abs_signal(input_in_window)
            elif self.model_params['input_type'] == 'spikes':
                input_in_window = self.reduce_dimension(input_in_window)
            mip_pred = self.model.predict(input_in_window.reshape(1,-1,1), verbose = 0, use_multiprocessing = True)[0]
            self.reverbs_params['max_interval_start'], self.reverbs_params['max_interval_end'], self.reverbs_params['min_interval_between'] = mip_pred

            reverbs_in_window = self._detect_reverbs(self.convert_binary_to_timestamps(spikes_in_window, input_type = 'spikes'))
            reverbs_binary_pred_long[window_start:window_end] = self.convert_timestamps_to_binary(reverbs_in_window, input_type = 'reverbs', size = self.model_params['window_size'])

        reverbs_binary_pred_long[reverbs_binary_pred_long >= 1] = 1
        reverbs_pred = self.convert_binary_to_timestamps(reverbs_binary_pred_long, input_type = 'reverbs')
        valid_reverbs = []
        params_multiplier = 10

        for reverb in reverbs_pred:
            if abs(reverb[0] - reverb[-1]) >  params_multiplier*self.reverbs_params['min_duration']:
                valid_reverbs.append([reverb[0], reverb[-1]])
        reverbs_pred = valid_reverbs
        #minIntervalBetween = self.bursts_params['min_interval_between']
        #self.bursts_params['min_interval_between'] = mip_pred[2]
        #reverbs_pred = self._detect_bursts(reverbs_pred)
        #self.bursts_params['min_interval_between'] = minIntervalBetween
        return reverbs_pred


    def normalize_signal(self, signal):
        return signal/abs(signal).max()
        
    def normalize_threshold(self, signal, threshold):
        return threshold/abs(signal).max()

    def reduce_dimension(self, X, input_type = None, reduction_factor = None): 
        
        if reduction_factor is None:
            reduction_factor = self.model_params['input_average']
            if reduction_factor is None:
                print('No reduction factor defined!')
                return 

        if self.model_params['input_type'] == 'spikes' or input_type == 'spikes': 
            if X.ndim == 1:
                X2 = np.zeros(len(X[::reduction_factor]))
                X2 = np.array([1 if len(np.where(X[reduction_factor*j:reduction_factor*j+reduction_factor] == 1)[0]) else 0 for j in range(len(X[::reduction_factor]))])
            else:    
                X2 = np.zeros((len(X), len(X[0][::reduction_factor])))
                for i, x in enumerate(X):
                    X2[i] = np.array([1 if len(np.where(x[reduction_factor*j:reduction_factor*j+reduction_factor] == 1)[0]) else 0 for j in range(len(x[::reduction_factor]))])
            return X2

        elif self.model_params['input_type'] == 'signal' or input_type == 'signal':
    
            numbers = X
            i = 0
            moving_averages = []

            while (i+1)*self.model_params['input_average'] < len(numbers):
                this_window = numbers[i*self.model_params['input_average'] : (i+1)*self.model_params['input_average']]
                window_average = sum(this_window) / self.model_params['input_average']
                moving_averages.append(window_average)
                i += 1

            if (i-1)*self.model_params['input_average'] < len(numbers):
                this_window = numbers[(i-1)*self.model_params['input_average'] :]
                window_average = sum(this_window) / len(this_window)
                moving_averages.append(window_average)
        
            return np.array(moving_averages)
                        

        else:
            print('No input type (signal or spikes) defined!')
            return


    def reduce_norm_abs_signal(self, signal):
        if signal.ndim == 1:
            return self.reduce_dimension(self.normalize_signal(abs(signal)), input_type = 'signal')
        else: 
            return np.array(list(map(self.reduce_dimension, self.normalize_signal(abs(signal)), ['signal' for _ in range(len(signal))])))



    def detect_net(self, inp, minChannelsParticipating = 8, minSimultaneousChannels = 6):
        
        if inp == 'reverbs':
            bursts_binary = self.reverbs_binary
        elif inp == 'bursts':
            bursts_binary = self.bursts_binary

        howMuchIsBursting = np.sum(bursts_binary, axis = 0)
        
        netBursts = []
        alreadyInBurst = False

        for i in range(len(howMuchIsBursting)):
            ## start net burst when at a timestamp there are more than minChannelParticipating bursting
            if (not alreadyInBurst) and (howMuchIsBursting[i] >= minChannelsParticipating):
                netBursts.append([i,i])
                alreadyInBurst = True
            ## when the number of channels bursting together drops below minSimultaneousChannels, end the net burst
            elif alreadyInBurst and howMuchIsBursting[i] < minSimultaneousChannels:
                alreadyInBurst = False
                netBursts[-1][-1] = i

        for k, netburst in enumerate(netBursts):
            ## starting at the beginning of every netburst, go against time (decrease timestamp)
            ## until less than 2 channels are bursting. Reassing the beginning time of the net burst
            ## to this timestamp
            startNetBurst, endNetBurst = netburst
            for i in range(startNetBurst, 0, -1):
                if(howMuchIsBursting[i] < 2):
                    netBursts[k][0] = i
                    break
            ## starting at the end of every netburst, go ahead in time (increase timestamp)
            ## until the last burst in the netburst group finishes. Assign this timestamp as
            ## the new end of netburst 
            for i in range(endNetBurst, len(howMuchIsBursting)):
                if(howMuchIsBursting[i] < 1):
                    netBursts[k][-1] = i
                    break
        
        netBursts_copy = []
        for i in range(1,len(netBursts)):
            if netBursts[i] != netBursts[i-1]:
                netBursts_copy.append(netBursts[i])
                
        netbin = np.zeros(len(bursts_binary[1]))
        for net in netBursts_copy:
            netbin[net[0]:net[1]+1] += 1
        netbin[netbin > 1] = 1 
        

        if inp == 'reverbs':
            self.net_reverbs_binary = netbin
            self.net_reverbs = self.convert_binary_to_timestamps(self.net_reverbs_binary, input_type = 'reverbs')
        elif inp == 'bursts':
            self.net_bursts_binary = netbin
            self.net_bursts = self.convert_binary_to_timestamps(self.net_bursts_binary, input_type = 'bursts')


    def save_spikes(self):
        columns_spikes = ['Channel ID', 'Channel Label', 'Well ID', 'Well Label', 'Compound ID', 'Compound Name']
        columns_spikes.extend(['Experiment', 'Dose [pM]', 'Dose Label'])
        columns_spikes.extend(['timestamp'])
        spikes_data_file = os.path.join(self.output_folder, f'{self.output_name}_REVERBS_PREDICTED.csv')

        

    def analyze_dataset(self, file = None, mode = 'csv', save_default = True):

        ## create output folder to save analysis
        output_folder_exists = os.path.exists(self.output_folder)
        if not output_folder_exists:
           os.makedirs(self.output_folder)

        print('--- Running full analysis ---\n')

        ### create dataframes to save reverbs, bursts, network bursts, and statistics
 
        columns_spikes = ['Channel ID', 'Channel Label', 'Well ID', 'Well Label', 'Compound ID', 'Compound Name']
        columns_spikes.extend(['Experiment', 'Dose [pM]', 'Dose Label', 'Timestamp'])
    

        columns_reverbs = ['Channel ID', 'Channel Label', 'Well ID', 'Well Label', 'Compound ID', 'Compound Name']
        columns_reverbs.extend(['Experiment', 'Dose [pM]', 'Dose Label'])
        columns_reverbs.extend(['Start timestamp [\u03BCs]', 'Duration [\u03BCs]', 'Spike Count', 'Spike Frequency [Hz]'])

        columns_bursts = ['Channel ID', 'Channel Label', 'Well ID', 'Well Label', 'Compound ID', 'Compound Name']
        columns_bursts.extend(['Experiment', 'Dose [pM]', 'Dose Label'])
        columns_bursts.extend(['Start timestamp [\u03BCs]', 'Duration [\u03BCs]', 'Spike Count', 'Spike Frequency [Hz]'])

        columns_net = ['Well ID', 'Well Label', 'Compound ID', 'Compound Name']
        columns_net.extend(['Experiment', 'Dose [pM]', 'Dose Label'])
        columns_net.extend(['Start timestamp [\u03BCs]', 'Duration [\u03BCs]', 'Spike Count', 'Spike Frequency [Hz]'])


        columns_stats = ['Filename', 'Well Label', 'Number of channels', 'Total number of spikes', 'Mean Firing Rate [Hz]']
        columns_stats.extend(['Stray spikes (%)'])
        columns_stats.extend(['Total number of networks bursts', 'Mean Network Bursting Rate [bursts/minute]', 'Mean Network Burst Duration [ms]'])
        columns_stats.extend(['NIBI', 'CV of NIBI'])
        columns_stats.extend(['Mean reverb per burst', 'Median of reverb per burst'])
        columns_stats.extend(['Mean net reverb per net burst', 'Median of net reverb per net burst', 'Total number of network reverb'])
        columns_stats.extend(['Mean net reverb frequency [reverb/min]', 'Mean net reverb duration [ms]', 'Mean in-reverb freq [Hz]'])

        spikes_data = []
        spikes_data.append(columns_spikes)

        reverbs_data_pred = []
        bursts_data_pred = []
        netBursts_data_pred = []
        stats_data_pred = []
        
        reverbs_data_pred.append(columns_reverbs)
        bursts_data_pred.append(columns_reverbs)
        netBursts_data_pred.append(columns_net)
        stats_data_pred.append(columns_stats)

        reverbs_data_def = []
        bursts_data_def = []
        netBursts_data_def = []
        stats_data_def = []

        reverbs_data_def.append(columns_reverbs)
        bursts_data_def.append(columns_reverbs)
        netBursts_data_def.append(columns_net)
        stats_data_def.append(columns_stats)


        self.loadmodel()
        print('--- Using model ', self.model_params['name'], ' ---\n')

        if mode == 'csv': 
            self.files_and_well_csv(file)


        for file_index, filename in enumerate(self.dataset):

            self.dataset_filename = filename 
            self.dataset_index = file_index 

            print('\n Analyzing dataset: ', filename, '\n')

            self.loadh5(filename)
            fullSignal = self.signal
            ### are these needed?
            uniqueWells = np.unique(self.wellsFromData)
            numberOfWells = np.unique(self.wellsFromData).shape[0]
            timestamp = np.arange(len(self.signal[0]))/self.samplingFreq
            total_seconds = len(timestamp)//self.samplingFreq
            total_minutes = total_seconds//60

            wellIDsToUse = [self.wellLabelIndexDict[label] for label in self.wellsLabels[file_index]]
            self.wellsIDs.append(wellIDsToUse)

            for well in wellIDsToUse:

                self.well = well
                
                print('Well: ', self.wellIndexLabelDict[well])

                signal = fullSignal[np.where(self.wellsFromData == well)[0]]
                self.loadsignal(signal)
                self.detect_threshold()
                self.detect_spikes()
    
                if save_default:

                    self.detect_reverbs(method = 'default')
                    self.detect_bursts()
                    self.detect_net('reverbs')
                    self.detect_net('bursts')

                    default_reverbs = self.reverbs 
                    default_reverbs_binary = self.reverbs_binary 
                    default_net_reverbs = self.net_reverbs 
                    default_net_reverbs_binary = self.net_reverbs_binary 
                    default_bursts = self.bursts 
                    default_bursts_binary = self.bursts_binary 
                    default_net_bursts = self.net_bursts 
                    default_net_bursts_binary = self.net_bursts_binary 

                self.detect_reverbs(method = 'model')
                self.detect_bursts()
                self.detect_net('reverbs')
                self.detect_net('bursts')
                pred_reverbs = self.reverbs 
                pred_reverbs_binary = self.reverbs_binary 
                pred_net_reverbs = self.net_reverbs 
                pred_net_reverbs_binary = self.net_reverbs_binary 
                pred_bursts = self.bursts  
                pred_bursts_binary = self.bursts_binary  
                pred_net_bursts = self.net_bursts
                pred_net_bursts_binary = self.net_bursts_binary

                ########################
                ### predicted
                ########################
                compoundID = 'No Compound'
                compoundName = 'No Compound'
                experiment = filename
                dose = 0
                doseLabel = 'Control'
                i = 0
                wellID = well
                commonToRow = [wellID, self.wellIndexLabelDict[wellID], compoundID, compoundName, experiment, dose, doseLabel]      
                for burst in pred_net_bursts:
                    startTime = int(burst[0]*100)
                    duration = int((burst[-1]-burst[0])*100)
                    spikes = self.spikes_binary
                    number = self.util.number_of_spikes_inside_burst(self.spikes_binary, burst)
                    freq = self.util.mean_innetburst_frequency(spikes, [burst])
                    newRow = commonToRow.copy()
                    newRow.extend([startTime, duration, number, freq])
                    netBursts_data_pred.append(newRow)

                for channel in range(np.where(self.wellsFromData == well)[0].shape[0]):
                    channelID = self.infoChannel[np.where(self.wellsFromData == well)[0][channel]]['ChannelID']
                    channelLabel = str(self.infoChannel[np.where(self.wellsFromData == well)[0][channel]]['Label'])[2:-1]
                    wellID = well

                    commonToRow = [channelID, channelLabel, wellID, self.wellIndexLabelDict[wellID], 
                                   compoundID, compoundName, experiment, dose, doseLabel]  
                    
                    for spike in self.spikes[channel]:
                        newRow = commonToRow.copy()
                        newRow.extend([spike])
                        spikes_data.append(newRow)

                    for burst in pred_reverbs[channel]:
                        startTime = int(burst[0]*100)
                        duration = int((burst[-1]-burst[0])*100)
                        spikes = self.spikes_binary[channel].reshape(-1,1).T
                        number = self.util.number_of_spikes_inside_burst(spikes, burst)
                        freq = self.util.mean_innetburst_frequency(spikes, [burst])
                        newRow = commonToRow.copy()
                        newRow.extend([startTime, duration, number, freq])
                        reverbs_data_pred.append(newRow)

                    for burst in pred_bursts[channel]:
                        startTime = int(burst[0]*100)
                        duration = int((burst[-1]-burst[0])*100)
                        spikes = self.spikes_binary[channel].reshape(-1,1).T
                        number = self.util.number_of_spikes_inside_burst(spikes, burst)
                        freq = self.util.mean_innetburst_frequency(spikes, [burst])
                        newRow = commonToRow.copy()
                        newRow.extend([startTime, duration, number, freq])
                        bursts_data_pred.append(newRow)
                        
                        
                newRow = [filename, self.wellIndexLabelDict[wellID], np.where(self.wellsFromData == well)[0].shape[0]]   
                newRow.extend([self.util.total_number_of_binary(self.spikes_binary)])
                newRow.extend([self.util.mean_firing_rate(self.spikes_binary, total_seconds = self.total_timesteps_signal//self.samplingFreq)])

                newRow.extend([self.util.random_spikes_percentage_net(self.spikes_binary, pred_net_bursts)]) # net
                newRow.extend([self.util.total_number_of_netBursts(pred_net_bursts)]) # net
                newRow.extend([self.util.mean_netbursting_rate(pred_net_bursts, total_minutes=self.total_timesteps_signal//self.samplingFreq//60)]) # net
                newRow.extend([self.util.mean_netburst_duration(pred_net_bursts)]) # net

                newRow.extend([self.util.mean_interNetBurstTrain_interval(pred_net_bursts)])
                newRow.extend([self.util.coeff_variance_interNetBurstTrain_interval(pred_net_bursts)])
                newRow.extend([self.util.mean_bursts_per_burstTrain(pred_reverbs, pred_bursts)])
                newRow.extend([self.util.median_bursts_per_burstTrain(pred_reverbs, pred_bursts)])
                newRow.extend([self.util.mean_bursts_per_burstTrain(pred_net_reverbs, pred_net_bursts, net = True)])
                newRow.extend([self.util.median_bursts_per_burstTrain(pred_net_reverbs, pred_net_bursts, net = True)])

                newRow.extend([self.util.total_number_of_netBursts(pred_net_reverbs)]) # net
                newRow.extend([self.util.mean_netbursting_rate(pred_net_reverbs, total_minutes=self.total_timesteps_signal//self.samplingFreq//60)]) # net
                newRow.extend([self.util.mean_netburst_duration(pred_net_reverbs)]) # net
                newRow.extend([self.util.mean_innetburst_frequency(self.spikes_binary, pred_net_reverbs)]) # net

                stats_data_pred.append(newRow)

                if save_default:
                    ########################
                    ### default
                    ########################
                    compoundID = 'No Compound'
                    compoundName = 'No Compound'
                    experiment = filename
                    dose = 0
                    doseLabel = 'Control'
                    i = 0
                    wellID = well
                    commonToRow = [wellID, self.wellIndexLabelDict[wellID], compoundID, compoundName, experiment, dose, doseLabel]      
                    for burst in default_net_bursts:
                        startTime = int(burst[0]*100)
                        duration = int((burst[-1]-burst[0])*100)
                        spikes = self.spikes_binary
                        number = self.util.number_of_spikes_inside_burst(self.spikes_binary, burst)
                        freq = self.util.mean_innetburst_frequency(spikes, [burst])
                        newRow = commonToRow.copy()
                        newRow.extend([startTime, duration, number, freq])
                        netBursts_data_def.append(newRow)

                    for channel in range(np.where(self.wellsFromData == well)[0].shape[0]):
                        channelID = self.infoChannel[np.where(self.wellsFromData == well)[0][channel]]['ChannelID']
                        channelLabel = str(self.infoChannel[np.where(self.wellsFromData == well)[0][channel]]['Label'])[2:-1]
                        wellID = well

                        commonToRow = [channelID, channelLabel, wellID, self.wellIndexLabelDict[wellID], 
                                       compoundID, compoundName, experiment, dose, doseLabel]  
                        for burst in default_reverbs[channel]:
                            startTime = int(burst[0]*100)
                            duration = int((burst[-1]-burst[0])*100)
                            spikes = self.spikes_binary[channel].reshape(-1,1).T
                            number = self.util.number_of_spikes_inside_burst(spikes, burst)
                            freq = self.util.mean_innetburst_frequency(spikes, [burst])
                            newRow = commonToRow.copy()
                            newRow.extend([startTime, duration, number, freq])
                            reverbs_data_def.append(newRow)


                        for burst in default_bursts[channel]:
                            startTime = int(burst[0]*100)
                            duration = int((burst[-1]-burst[0])*100)
                            spikes = self.spikes_binary[channel].reshape(-1,1).T
                            number = self.util.number_of_spikes_inside_burst(spikes, burst)
                            freq = self.util.mean_innetburst_frequency(spikes, [burst])
                            newRow = commonToRow.copy()
                            newRow.extend([startTime, duration, number, freq])
                            bursts_data_def.append(newRow)
                            
                            
                    newRow = [filename, self.wellIndexLabelDict[wellID], np.where(self.wellsFromData == well)[0].shape[0]]   
                    newRow.extend([self.util.total_number_of_binary(self.spikes_binary)])
                    newRow.extend([self.util.mean_firing_rate(self.spikes_binary, total_seconds = self.total_timesteps_signal//self.samplingFreq)])

                    newRow.extend([self.util.random_spikes_percentage_net(self.spikes_binary, default_net_bursts)])
                    newRow.extend([self.util.total_number_of_netBursts(default_net_bursts)])
                    newRow.extend([self.util.mean_netbursting_rate(default_net_bursts, total_minutes=self.total_timesteps_signal//self.samplingFreq//60)])
                    newRow.extend([self.util.mean_netburst_duration(default_net_bursts)])

                    newRow.extend([self.util.mean_interNetBurstTrain_interval(default_net_bursts)])
                    newRow.extend([self.util.coeff_variance_interNetBurstTrain_interval(default_net_bursts)])
                    newRow.extend([self.util.mean_bursts_per_burstTrain(default_reverbs, default_bursts)])
                    newRow.extend([self.util.median_bursts_per_burstTrain(default_reverbs, default_bursts)])
                    newRow.extend([self.util.mean_bursts_per_burstTrain(default_net_reverbs, default_net_bursts, net = True)])
                    newRow.extend([self.util.median_bursts_per_burstTrain(default_net_reverbs, default_net_bursts, net = True)]) # net reverbs per net bursts
             
                    newRow.extend([self.util.total_number_of_netBursts(default_net_reverbs)])
                    newRow.extend([self.util.mean_netbursting_rate(default_net_reverbs, total_minutes=self.total_timesteps_signal//self.samplingFreq//60)])
                    newRow.extend([self.util.mean_netburst_duration(default_net_reverbs)])
                    newRow.extend([self.util.mean_innetburst_frequency(self.spikes_binary, default_net_reverbs)])
                    
                    stats_data_def.append(newRow)

                      
        ########################
        ### save files
        ########################      

        spikes_data_file = os.path.join(self.output_folder, f'{self.output_name}_SPIKES.csv')
        if self.analysis_params['save_spikes']: pd.DataFrame(spikes_data[1:], columns = spikes_data[0]).to_csv(spikes_data_file, index = False)

        reverbs_data_pred_file = os.path.join(self.output_folder, f'{self.output_name}_REVERBS_PREDICTED.csv')
        bursts_data_pred_file = os.path.join(self.output_folder, f'{self.output_name}_BURSTS_PREDICTED.csv')
        netBursts_data_pred_file = os.path.join(self.output_folder, f'{self.output_name}_NET_BURSTS_PREDICTED.csv')
        stats_data_pred_file = os.path.join(self.output_folder, f'{self.output_name}_STATS_PREDICTED.csv')

        if self.analysis_params['save_reverbs']    : pd.DataFrame(reverbs_data_pred[1:], columns = reverbs_data_pred[0]).to_csv(reverbs_data_pred_file, index = False)
        if self.analysis_params['save_bursts']     : pd.DataFrame(bursts_data_pred[1:], columns = bursts_data_pred[0]).to_csv(bursts_data_pred_file, index = False)
        if self.analysis_params['save_net_bursts'] : pd.DataFrame(netBursts_data_pred[1:], columns = netBursts_data_pred[0]).to_csv(netBursts_data_pred_file, index = False)
        if self.analysis_params['save_stats']      : pd.DataFrame(stats_data_pred[1:], columns = stats_data_pred[0]).to_csv(stats_data_pred_file, index = False)


        if save_default:            
            reverbs_data_def_file = os.path.join(self.output_folder, f'{self.output_name}_REVERBS_DEFAULT.csv')
            bursts_data_def_file = os.path.join(self.output_folder, f'{self.output_name}_BURSTS_DEFAULT.csv')
            netBursts_data_def_file = os.path.join(self.output_folder, f'{self.output_name}_NET_BURSTS_DEFAULT.csv')
            stats_data_def_file = os.path.join(self.output_folder, f'{self.output_name}_STATS_DEFAULT.csv')

            if self.analysis_params['save_reverbs']    : pd.DataFrame(reverbs_data_def[1:], columns = reverbs_data_def[0]).to_csv(reverbs_data_def_file, index = False)
            if self.analysis_params['save_bursts']     : pd.DataFrame(bursts_data_def[1:], columns = bursts_data_def[0]).to_csv(bursts_data_def_file, index = False)
            if self.analysis_params['save_net_bursts'] : pd.DataFrame(netBursts_data_def[1:], columns = netBursts_data_def[0]).to_csv(netBursts_data_def_file, index = False)
            if self.analysis_params['save_stats']      : pd.DataFrame(stats_data_def[1:], columns = stats_data_def[0]).to_csv(stats_data_def_file, index = False)
                    


        print('\n--- Done! ---')

    ##########################################################################################################
    ##########################################################################################################
    # plotting functions
    ########
    ########

    def plot_window(self, signal, start_time = None, duration = None, threshold = None, spikes = None, reverberations = None, bursts = None, net_bursts = None, save = False, output_name = None):
        
        ##########################################################################################################
        # check if variables exist, have correct shape, and select slices to plot based on start_time and duration
        ########

        if signal.ndim > 1:
            print("More than one signal (channel) input!")
            return

        if start_time is None: start_time = 0
        if duration is None: duration = int(len(signal)/self.samplingFreq)

        start_timestamp = int(start_time*self.samplingFreq)
        
        if start_timestamp > len(signal):
            print("Start time (in seconds) bigger than recorded signal time.")
            return

        duration_ts = int(duration*self.samplingFreq)
        print(start_timestamp, duration_ts)
        if start_timestamp + duration_ts > len(signal):
            print("Start time + duration (in seconds) bigger than recorded signal time.")
            return

        end_timestamp = start_timestamp + duration_ts

        spikes_to_plot = []
        if spikes:
            if isinstance(spikes, np.ndarray) and signal.ndim == 1:
                print('More than one spike channel input!')
                return
            elif self.util._has_list(spikes):
                print('More than one spike channel input!')
                return
            spikes = np.array(spikes)
            spikes_to_plot = spikes[(spikes >= start_timestamp) & (spikes <= end_timestamp)]


        def select_bursts_to_plot(bursts, start_timestamp, end_timestamp):
            bursts = np.array(bursts)
            bursts_starts = bursts.T[0]
            bursts_ends = bursts.T[1]
            try:
                plot_from = np.where(bursts_starts >= start_timestamp)[0][0] - 1
                plot_to = np.where(bursts_ends <= end_timestamp)[0][-1] + 1
                if plot_from <= 0: plot_from = 0
                if plot_to >= len(bursts): plot_to = len(bursts)
                return bursts[plot_from:plot_to]
            except:
                return np.array([])
            
        reverbs_to_plot = []
        if reverberations:
            if isinstance(reverberations, np.ndarray):
                if reverberations.ndim > 2:
                    print("More than one reverberations channel input!")
                    return
            elif self.util._has_list(reverberations[0]):
                    print("More than one reverberations channel input!")
                    return            
            reverbs_to_plot = select_bursts_to_plot(reverberations, start_timestamp, end_timestamp)
            
        
        bursts_to_plot = []
        if bursts:
            if isinstance(bursts, np.ndarray):
                if bursts.ndim > 2:
                    print("More than one reverberations channel input!")
                    return
            elif self.util._has_list(bursts[0]):
                    print("More than one reverberations channel input!")
                    return            
            bursts_to_plot = select_bursts_to_plot(bursts, start_timestamp, end_timestamp)

        net_bursts_to_plot = []
        if net_bursts:
            if isinstance(net_bursts, np.ndarray):
                if net_bursts.ndim > 2:
                    print("More than one reverberations channel input!")
                    return
            elif self.util._has_list(net_bursts[0]):
                    print("More than one reverberations channel input!")
                    return            
            net_bursts_to_plot = select_bursts_to_plot(net_bursts, start_timestamp, end_timestamp)
        ########
        ##########################################################################################################
        norm_signal = self.normalize_signal(signal)

        fig = plt.figure(figsize=(12,12))

        for spike in spikes_to_plot:
            plt.axvline(spike/self.samplingFreq,0,0.025, c = '#660033', lw = 1)

        for reverb in reverbs_to_plot:
            plt.hlines(-1.2,reverb[0]/self.samplingFreq,reverb[-1]/self.samplingFreq, colors = 'blue', lw = 7)

        for burst in bursts_to_plot:
            plt.hlines(-1.15,burst[0]/self.samplingFreq,burst[-1]/self.samplingFreq, colors = '#FF3333', lw = 7)

        for net_bursts in net_bursts_to_plot:
            plt.hlines(-1.1,net_bursts[0]/self.samplingFreq,net_bursts[-1]/self.samplingFreq, colors = 'orange', lw = 3)

        plt.plot(np.arange(start_timestamp, end_timestamp)/self.samplingFreq, norm_signal[start_timestamp:end_timestamp], c = 'k', lw = 1)
        # plt.axhline(0, c = 'k', lw = 1)
        plt.axhline(self.normalize_threshold(signal, threshold), color = 'k', lw = 2)
        plt.axhline(-self.normalize_threshold(signal, threshold), color = 'k', lw = 2)
        plt.xlabel('Time [s]')
        plt.ylim(-1.3,1.3)
        plt.xlim(start_time, end_timestamp/self.samplingFreq)
        plt.grid(ls = 'dotted')
        if save:
            plt.savefig(f'{output_name}.jpg')
        plt.show()
        plt.close()


    def plot_raster(self, spikes, reverbs = None, bursts = None, net_reverbs = None, net_bursts = None):
        
        if self.util._has_list(spikes):
            number_of_channels = len(spikes)
        else:
            number_of_channels = 1
            spikes = [spikes]
            
        def plot_channel_(bursts):
            for channel, bursts_channel in enumerate(bursts):
                for burst in bursts_channel:
                    y1 = (height*(number_of_channels-channel-1))+(0.9*height)
                    y2 = (height*(number_of_channels-channel))-(0.9*height)
                    plt.fill_between(np.array([burst[0]/self.samplingFreq, burst[1]/self.samplingFreq]), 
                                     y1, y2, color = 'r', alpha = 0.3, zorder = 1)
        def plot_well_(nets):
            for net in nets:
                plt.fill_between(np.array([net[0]/self.samplingFreq, net[1]/self.samplingFreq]), 
                    1, 0, color = 'k', alpha = 0.3, zorder = 1)
                
        plt.figure(figsize = (10,5))
        height = 1/number_of_channels
        y_ticks = np.linspace(0,1,number_of_channels+1)[1:]-(height/2)
        plt.hlines(y_ticks, 0, self.total_timesteps_signal/self.samplingFreq, color = 'k', lw = 1)
        for channel, spikes_c in enumerate(spikes):
            plt.vlines(np.array(spikes_c)/self.samplingFreq,(height*(number_of_channels-channel-1))+(height/4), (height*(number_of_channels-channel))-(height/4), color = 'k', lw = 1)
        
        if reverbs: plot_channel_(reverbs)
        if net_bursts: plot_well_(net_bursts)
        plt.ylim(0,1)
        plt.xlim(0, self.total_timesteps_signal/self.samplingFreq)
        plt.tick_params(axis = 'y',
                        which = 'both',
                        direction = 'in')
        plt.yticks(y_ticks[::-1], labels = [f'e{channel+1}' for channel in range(number_of_channels)], fontsize = 20)
        plt.xticks([])
        plt.show()


    def plot_raster_well(self, file:str, well, method = 'default', reverbs = False, bursts = False, net_reverbs = False, net_bursts = False):

        self.loadwell(file, well, 
                      method = method, spikes = True, 
                      reverbs = reverbs, bursts = bursts,
                      net_reverbs = net_reverbs, net_bursts = net_bursts)

        reverbs_ = self.reverbs if reverbs else None
        bursts_ = self.bursts if bursts else None
        net_reverbs_ = self.net_reverbs if net_reverbs else None
        net_bursts_ = self.net_bursts if net_bursts else None

        self.plot_raster(self.spikes, reverbs = reverbs_, bursts = bursts_,
                        net_reverbs = net_reverbs_, net_bursts = net_bursts_)
        


        

