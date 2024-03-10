import pandas as pd
import os
from contextlib import redirect_stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with redirect_stderr(open(os.devnull, "w")):
    import tensorflow as tf
import h5py
import numpy as np 
import matplotlib.pyplot as plt
from automea import util

class Analysis:
    """
    Class for performing analysis of MEA datasets.

    This class provides methods for loading datasets, performing various analyses,
    and visualizing the results.


    Attributes
    ----------
    util : module
        Utility functions module.
    dataset : list
        The dataset. Default is an empty list.
    dataset_filename : str
        The filename of the dataset. Default is an empty string.
    dataset_index : int or None
        The index of the dataset. Default is None.
    path_to_dataset : str
        The path to the dataset. Default is an empty string.
    path_to_csv : str
        The path to the CSV file. Default is an empty string.
    path_to_model : str
        The path to the model file. Default is an empty string.
    output_name : str
        The name of the output. Default is an empty string.
    output_folder : str
        The output folder. Default is an empty string.
    wellsIDs : list
        The IDs of the wells. Default is an empty list.
    wellsLabels : list
        The labels of the wells. Default is an empty list.
    well : str or None
        The selected well. Default is None.
    model : object or None
        The ML model. Default is None.
    model_name : str or None
        The name of the model. Default is None.
    signal : array_like or None
        The recorded signal. Default is None.
    time : array_like or None
        The time array. Default is None.
    spikes : array_like or None
        The detected spikes. Default is None.
    spikes_binary : array_like or None
        The binary representation of detected spikes. Default is None.
    reverbs : array_like or None
        The detected reverberations. Default is None.
    reverbs_binary : array_like or None
        The binary representation of detected reverberations. Default is None.
    bursts : array

    Methods
    -------
    files_and_well_csv(file)
        Load filenames and associated wells from a CSV file.
    
    loadmodel(setting = 'default')
        Load a machine learning model to perform reverberations/bursts detection.

    loadsignal(signal)
        Load a signal into the analysis object.

    loadh5(filename = None)
        Load data from an HDF5 file into the analysis object.

    convert_signal(signal, adzero, conversionfactor, exponent)
        Convert raw signal values to physical units using the provided parameters:
        adzero, conversionfactor, and exponent

    convert_threshold(threshold, adzero, conversionfactor, exponent)
        Convert threshold value to physical units using the provided parameters:
        adzero, conversionfactor, and exponent.

    loadspikes(spikes)
        Load spike data into the analysis object.

    loadwell(file:str, well, method = 'default', spikes = False, reverbs = False, 
                 bursts = False, net_reverbs = False, net_bursts = False)
        Load data for a specific well.

    detect_threshold()
        Detect threshold value for spike detection.

    convert_timestamps_to_binary(input_timestamp, input_type, size = None)
        Convert timestamps to binary representation.

    convert_binary_to_timestamps(input_binary, input_type)
        Convert binary representation data to timestamps.

    _convert_binary_reverb_to_timestamp(input_binary)
        Convert binary representation of reverbs to timestamps.

    _detect_spikes(signal, threshold)
        Detect spikes in a signal based on a given threshold value.

    detect_spikes()
        Detect spikes in the signal.

    reverbs_params_default()
        Set default parameters for reverberations/bursts detection.

    reverbs_params_manual(params)
        Set manual parameters for reverberations detection.

    detect_reverbs(method = 'default')
        Detects reverberations based on the specified method.

    _detect_reverbs(spikes)
        Detect reverberations based on spikes timestamps.

    detect_bursts()
        Detect bursts in the signal based on the detected reverberations.

    _detect_bursts(reverbs)
        Detect bursts based on reverberations.

    _predict_reverbs(model_input, spikes_binary)
        Predict reverberations using a pre-trained machine learning model.

    normalize_signal(signal)
        Normalize the input signal.

    normalize_threshold(signal, threshold)
        Normalize the threshold relative to the input signal.
    
    reduce_dimension(X, input_type = None, reduction_factor = None)
        Reduce the dimensionality of the input data.

    reduce_norm_abs_signal(signal)
        Reduce the dimensionality of the normalized absolute signal.

    detect_net(inp, minChannelsParticipating=8, minSimultaneousChannels=6)
        Detect net bursts or net reverbs.

    save_spikes()
        Save spikes data to a CSV file.

    analyze_dataset(file=None, mode='csv', save_default=False)
        Analyze the datasets from a CSV file,  and save the results to CSV files.

    plot_window(signal, start_time=None, duration=None, threshold=None, spikes=None, reverberations=None,
                bursts=None, net_bursts=None, save=False, output_name=None, figsize=(6, 6), yunits='a.u.',
                xunits='s')
        Plot a window of the signal with detected spikes, reverberations, bursts, and network bursts.

    plot_raster(spikes, reverbs = None, bursts = None, net_reverbs = None, net_bursts = None)
        Plot a raster plot of spikes with optional overlay of reverberations, bursts, and network bursts.
    
    plot_raster_well(file:str, well, method = 'default', reverbs = False, bursts = False, 
                     net_reverbs = False, net_bursts = False)
        Plot a raster plot for a specific well with optional overlay of events.
    """

    def __init__(self):
        """
        Initializes class with default attributes.
        """

        # import util functions
        self.util = util

        # initialize attributes to empty list/ empty string / None
        self.dataset = []
        self.dataset_filename = ''
        self.dataset_index = None
        self.path_to_dataset = ''
        self.path_to_csv = ''
        self.path_to_model = ''
        self.output_name = ''
        self.output_folder = ''
        self.wellsIDs = []
        self.wellsLabels = []
        self.well = None
        self.model = None
        self.model_name = None
        self.signal = None
        self.time = None
        self.spikes = None
        self.spikes_binary = None
        self.reverbs = None
        self.reverbs_binary = None
        self.bursts = None
        self.bursts_binary = None
        self.plot_name = None


        self.samplingFreq = 10_000 # default sampling frequency = 10kHZ

        self.total_timesteps_signal = 6_000_000 #default number of timesteps in one recording - 10min = 60s (for sampl freq 10kHz) = 6million points

        self.time = np.linspace(0, (self.total_timesteps_signal-1)/self.samplingFreq, self.total_timesteps_signal) # time array in seconds

        # parameters used to detect threshold
        self.threshold_params = {'rising'      : 5, # how many standard deviations are used to set threshold
                                'startTime'    : 0, # start time to consider signal
                                'baseTime'     : 0.250, # in ms, for how long to consider signal for std calculation 
                                'segments'     : 10} # how many segment to perform calculation

        self.spikes_params = {'deadtime': 3_000*1e-6} # time after detecting a spike for which spike detection is halted

        # parameters used for reverberations / bursts detection - Max Interval parameters
        self.reverbs_params = {'max_interval_start'   : 15, 
                              'max_interval_end'     : 20,
                              'min_interval_between' : 25,
                              'min_duration'         : 20,
                              'min_spikes'           : 5}

        # parameter to merge reverberations into bursts
        self.bursts_params = {'min_interval_between'   : 300}

        # pretained ML models for burst detection distributed with the package
        self._pretrained_models = ['signal30.h5']

        # parameters for model based burst detection
        self.model_params = {'name'           : None, # name of the model
                             'input_type'     : None, # type of input: signal or spikes
                             'input_average'  : 30,   # how many points is used to calculate average of input
                             'window_size'    : 50_000, # size of window (in timestamps) used as input for the model (before averaging)
                             'window_overlap' : 25_000} # overlap between windows when sweeping a channel

        # parameters for analysis - which quantities user wants to save - each one creates an output file
        self.analysis_params = {'save_spikes'      : False,
                                'save_reverbs'     : False,
                                'save_bursts'      : False,
                                'save_net_reverbs' : False,
                                'save_net_bursts'  : False,
                                'save_stats'       : False}

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
        self.wellIndexLabelDict = dict(zip(keys, items)) # dictionary to convert well-index to well-label
        self.wellLabelIndexDict = dict(zip(items, keys)) # dictionary to convert well-label to well-index

    def files_and_well_csv(self, file):
        """
        Load filenames and associated wells from a CSV file.

        This method reads a CSV file containing filenames and associated wells,
        and populates the `dataset` and `wellsLabels` attributes accordingly.

        Parameters
        ----------
        file : str
            The name of the CSV file to load, relative to the path_to_csv attribute.

        Returns
        -------
        None

        Examples
        --------
        >>> obj = Analysis()
        >>> obj.files_and_well_csv('data.csv')

        CSV file format example:
        filename;wells
        file1.h5;A1,B2,C3
        file2.h5f;all
        file3.h5;D4,E5,F6

        In the above example, we want to analyze wells A1, B2, and C3 fromfile1.h5,
        all wells from file2.h5, and D4, E5 and F6 from and file3.h5.
        """

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
        """
        Load a machine learning model to perform reverberations/bursts detection.

        This method loads a machine learning model specified by `model_name` and updates the
        `model_params` attribute based on the chosen setting ('default' or 'manual'). If no
        model_name is provided, it loads a default model named 'signal30.h5'.

        Parameters
        ----------
        setting : str, optional
            The setting to use for loading the model. Options are 'default' or 'manual'. 

        Returns
        -------
        None

        Examples
        --------
        >>> obj = Analysis()
        >>> obj.loadmodel()  # Load default model with default settings

        >>> obj.loadmodel(setting='manual')  # Load default model with manual settings

        Notes
        -----
        - If `model_name` is already specified, the method updates `model_params` accordingly.
        - The default settings include setting `input_average` to 30, `window_size` to 50,000, 
          and `window_overlap` to 25,000.
        - If a model with the same name as `model_name` is found in `_pretrained_models`, 
          it sets `input_type` based on the model name and updates other parameters accordingly.

        """

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
        """
        Load a signal into the analysis object.

        Parameters
        ----------
        signal : array_like
            The signal data to be loaded for analysis. It can be a 1D or 2D array,
            depending if the signal has one or multiple channels.

        Returns
        -------
        None

        Notes
        -----
        - If the signal is 1D, it is assumed to be a one-channel time series signal.
        - If the signal is 2D, it is assumed to be a collection of channels, with signals over time.
        - The `total_timesteps_signal` attribute is updated to reflect the length of the signal.
        - The `time` attribute is generated based on the sampling frequency and the length of the signal.

        Examples
        --------
        >>> obj = Analysis()
        >>> obj.loadsignal(signal_data)

        """

        self.signal = signal
        if np.array(signal.ndim) == 1:
            self.total_timesteps_signal = len(signal)
        else:
            self.total_timesteps_signal = np.array(signal).shape[1]
        self.time = np.linspace(0, (self.total_timesteps_signal-1)/self.samplingFreq, self.total_timesteps_signal)


    def loadh5(self, filename = None):
        """
        Load data from an HDF5 file into the analysis object.

        Parameters
        ----------
        filename : str, optional
            The name of the HDF5 file to load. If None, it loads the dataset specified in the dataset attribute.

        Returns
        -------
        None

        Notes
        -----
        - If None, sets the dataset attribute equals to the filename input
        - The 'infoChannel', 'adZero', 'conversionFactor', 'exponent', and 'wellsFromData' attributes are updated
          based on the data loaded from the HDF5 file.

        Examples
        --------
        >>> obj = Analysis()
        >>> obj.loadh5()  # Load the dataset from dataset attribute

        >>> obj.loadh5('example.h5')  # Load data from a specific HDF5 file
        """
        
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
        if len(np.unique(self.adZero)): self.adZero = self.adZero[0][0]
        self.conversionFactor =  np.array([self.infoChannel[i][10] for i in range(len(self.infoChannel))]).reshape(len(self.infoChannel), 1)
        if len(np.unique(self.conversionFactor)): self.conversionFactor = self.conversionFactor[0][0]
        self.exponent = np.array([self.infoChannel[i][7] for i in range(len(self.infoChannel))]).reshape(len(self.infoChannel), 1)
        if len(np.unique(self.exponent)): self.exponent = self.exponent[0][0]
        self.wellsFromData = np.array([self.infoChannel[i][2] for i in range(len(self.infoChannel))]).reshape(len(self.infoChannel), 1)
        self.wellsFromData = self.wellsFromData.flatten()


    def convert_signal(self, signal, adzero, conversionfactor, exponent):
        """
        Convert raw signal values to physical units using the provided parameters:
        adzero, conversionfactor, and exponent

        Parameters
        ----------
        signal : array_like
            The raw signal values to be converted.

        adzero : float
            The AD Zero value.

        conversionfactor : float
            The conversion factor to convert raw values to physical units.

        exponent : float
            The exponent applied during conversion.

        Returns
        -------
        array_like
            The converted signal values in physical units.

        Examples
        --------
        >>> obj = Analysis()
        >>> converted_signal = obj.convert_signal(raw_signal, ad_zero_value, conversion_factor, exponent_value)
        """

        return 1e6*(signal-adzero)*conversionfactor*10.**exponent

    def convert_threshold(self, threshold, adzero, conversionfactor, exponent):
        """
        Convert threshold value to physical units using the provided parameters:
        adzero, conversionfactor, and exponent.

        Parameters
        ----------
        threshold : float
            The threshold value to be converted.

        adzero : float
            The AD Zero value.

        conversionfactor : float
            The conversion factor to convert raw values to physical units.

        exponent : float
            The exponent applied during conversion.

        Returns
        -------
        float
            The converted threshold value in physical units.

        Examples
        --------
        >>> obj = Analysis()
        >>> converted_threshold = obj.convert_threshold(threshold_value, ad_zero_value, conversion_factor, exponent_value)
        """

        return self.convert_signal(threshold, adzero, conversionfactor, exponent)


    def loadspikes(self, spikes):
        """
        Load spike data into the analysis object.

        This method allows loading spike data into the analysis object for further processing.

        Parameters
        ----------
        spikes : array_like
            The spike data to be loaded for analysis.

        Returns
        -------
        None
        """

        self.spikes = spikes


    def loadwell(self, file:str, well, method = 'default', spikes = False, reverbs = False, bursts = False, net_reverbs = False, net_bursts = False):
        """
        Load data for a specific well.

        This method loads data for a specific well from an HDF5 file into the analysis object for further processing.
        Optionally, it can also detect spikes and analyze them for reverberations and bursts, calling the deticated methods.

        Parameters
        ----------
        file : str
            The name of the HDF5 file containing the data.
        well : str or int
            The label or index of the well to load.
        method : str, optional
            The method to use for detecting reverbs and bursts if spikes are detected. Default is 'default'.
        spikes : bool, optional
            Whether to detect spikes. Default is False.
        reverbs : bool, optional
            Whether to analyze reverberations. Default is False.
        bursts : bool, optional
            Whether to analyze bursts. Default is False.
        net_reverbs : bool, optional
            Whether to analyze net reverberations. Default is False.
        net_bursts : bool, optional
            Whether to analyze net bursts. Default is False.

        Returns
        -------
        None

        Examples
        --------
        >>> obj = Analysis()
        >>> obj.loadwell('data.h5', 'well_label', method='model', spikes=True, reverbs=True)

        Notes
        -----
        - The `file` parameter specifies the name of the HDF5 file containing the data.
        - The `well` parameter can be either a well label (str) or index (int).
        - The `method` parameter determines the method to use for detecting reverberations and bursts.
          It is only used if spikes are detected and defaults to 'default'.
        - The `spikes`, `reverbs`, `bursts`, `net_reverbs`, and `net_bursts` parameters control which
          analyses to perform after loading the data. They default to False.
        """
        
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
        """
        Detect threshold value for spike detection.

        This method calculates the threshold(s) value(s) based on the signal data and threshold parameters
        provided in the analysis object.

        Returns
        -------
        None

        Examples
        --------
        >>> obj = Analysis()
        >>> obj.detect_threshold()

        Notes
        -----
        - If the 'signal' attribute is not a numpy array, an error message is printed, and the method returns.
        - The threshold value is calculated segment-wise based on the mean and standard deviation of the signal.
        - The calculated threshold value is stored in the 'threshold' attribute of the analysis object.
        """

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

    def convert_timestamps_to_binary(self, input_timestamp, input_type, size = None):
        """
        Convert timestamps to binary representation.

        This method converts timestamps to a binary representation suitable for the specified input type,
        such as 'spikes', 'reverbs', or 'bursts'.

        Parameters
        ----------
        input_timestamp : array_like
            The timestamps data to be converted to binary representation.
        input_type : str
            The type of input timestamps. Options are 'spikes', 'reverbs', or 'bursts'. 
        size : int, optional
            The size of the binary representation dta. Default is None.

        Returns
        -------
        array_like
            The binary representation of the timestamps data.

        Examples
        --------
        >>> obj = Analysis()
        >>> spikes_binary = obj.convert_timestamps_to_binary(spikes_timestamps, input_type='spikes')

        Notes
        -----
        - If 'size' is not provided, it defaults to the total number of timesteps in the signal.
        - For 'spikes', each spike timestamp is represented as 1 in the binary array.
        - For 'reverbs' and 'bursts', each timestamp range is represented as 1 in the binary array.
        """
        
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
        


    def convert_binary_to_timestamps(self, input_binary, input_type):
        """
        Convert binary representation data to timestamps.

        This method converts a binary representation to timestamps for the specified input type,
        such as 'spikes', 'reverbs', or 'bursts'.

        Parameters
        ----------
        input_binary : array_like
            The binary representation data to be converted to timestamps.
        input_type : str, optional
            The type of input binary. Options are 'spikes', 'reverbs', or 'bursts'.

        Returns
        -------
        list or list of lists
            The timestamps corresponding to the binary representation.

        Examples
        --------
        >>> obj = Analysis()
        >>> spikes_timestamps = obj.convert_binary_to_timestamps(spikes_binary, input_type='spikes')

        Notes
        -----
        - For 'spikes', each 1 in the binary array represents a spike timestamp.
        - For 'reverbs' and 'bursts', consecutive 1s in the binary array represent timestamp ranges.
        """

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
        """
        Convert binary representation of reverbs to timestamps.

        This method converts a binary representation of reverbs to timestamps.

        Parameters
        ----------
        input_binary : array_like
            The binary representation of reverbs.

        Returns
        -------
        list
            List of timestamp ranges corresponding to reverbs.

        Examples
        --------
        >>> obj = Analysis()
        >>> reverb_timestamps = obj._convert_binary_reverb_to_timestamp(reverb_binary)

        Notes
        -----
        - This method converts a binary representation of reverbs to a list of timestamp ranges.
        - Consecutive 1s in the binary array represent timestamp ranges for reverberations.
        """

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
        """
        Detect spikes in a signal based on a given threshold value.

        Parameters
        ----------
        signal : array_like
            The signal in which spikes are to be detected.
        threshold : float
            The threshold value for spike detection.

        Returns
        -------
        list
            List of spike timestamps.

        Examples
        --------
        >>> obj = Analysis()
        >>> spikes = obj._detect_spikes(signal_data, threshold_value)

        Notes
        -----
        - It uses a deadtime parameter to avoid detecting multiple spikes within a short time period.
        """

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
        """
        Detect spikes in the signal.

        This method detects spikes from the signal attribute of the analysis object based on the threshold attribute.

        Returns
        -------
        None


        Notes
        -----
        - It checks if the signal attribute is a numpy array, and if not, prints an error message and returns.
        - If the signal attribute is a 1D numpy array, it detects spikes and updates the spikes and spikes_binary attributes.
        - If the signal attribute is a 2D numpy array (multiple channels), it detects spikes for each channel and updates
          the spikes and spikes_binary attributes accordingly.
        """

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
        """
        Set default parameters for reverberations/bursts detection.

        Returns
        -------
        None

        Notes
        -----
        - This method sets default parameters for detecting reverberations, including:
          - max_interval_start: Maximum interval start time for a reverberation.
          - max_interval_end: Maximum interval end time for a reverberation.
          - min_interval_between: Minimum interval between reverberations.
          - min_duration: Minimum duration of a reverberation.
          - min_spikes: Minimum number of spikes required to consider a reverberation.
        """

        self.reverbs_params['max_interval_start'] = 15
        self.reverbs_params['max_interval_end'] = 20
        self.reverbs_params['min_interval_between'] = 25
        self.reverbs_params['min_duration'] = 20
        self.reverbs_params['min_spikes'] = 5


    def reverbs_params_manual(self, params):
        """
        Set manual parameters for reverberations detection.

        Parameters
        ----------
        params : list
            List containing manual parameters for reverberations detection in the following order:
            - max_interval_start: Maximum interval start time for a reverberation.
            - max_interval_end: Maximum interval end time for a reverberation.
            - min_interval_between: Minimum interval between reverberations.
            - min_duration: Minimum duration of a reverberation.
            - min_spikes: Minimum number of spikes required to consider a reverberation.

        Returns
        -------
        None
        """

        self.reverbs_params['max_interval_start'] = params[0]
        self.reverbs_params['max_interval_end'] = params[1]
        self.reverbs_params['min_interval_between'] = params[2]
        self.reverbs_params['min_duration'] = params[3]
        self.reverbs_params['min_spikes'] = params[4]



    def detect_reverbs(self, method = 'default'): 
        """
        Detects reverberations based on the specified method.

        Parameters
        ----------
        method : str, optional
            The method used for reverberations detection. Options are 'default', 'manual', or 'model'. Default is 'default'.

        Returns
        -------
        None

        Notes
        -----
        - If method is 'default', the default parameters for reverberations detection are used.
        - If method is 'manual', the user can specify manual parameters for reverberations detection.
        - If method is 'model', reverberations are detected using a pre-trained machine learning model.
        - Detect reverberations are stored in the reverb attribute.
        """

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
        """
        Detect reverberations based on spikes timestamps.

        Parameters
        ----------
        spikes : list
            List of spike timestamps.

        Returns
        -------
        list
            List of timestamp ranges corresponding to reverberations.
        
        Notes
        -----
        - Implements the Max Interval Method
        """

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
        """
        Detect bursts in the signal based on the detected reverberations.

        Returns
        -------
        None

        Notes
        -----
        - If no reverberations are detected or if the 'reverbs' attribute is empty, no bursts will be detected.
        - Detected bursts are stored in the 'bursts' attribute.
        """

        if self.util._has_list(self.spikes) is False:
            self.bursts = self._detect_bursts(self.reverbs)
        else:
            self.bursts = list(map(self._detect_bursts, self.reverbs))
            self.bursts_binary = np.zeros((len(self.bursts), self.total_timesteps_signal))
            for channel, bursts_ in enumerate(self.bursts):
                self.bursts_binary[channel] = self.convert_timestamps_to_binary(bursts_, input_type = 'bursts')
        

    def _detect_bursts(self, reverbs):
        """
        Detect bursts based on reverberations.

        Parameters
        ----------
        reverbs : list
            List of timestamp ranges corresponding to detected reverberations.

        Returns
        -------
        list
            List of timestamp ranges corresponding to detected bursts.

        Notes
        -----
        - This method iteratively merges adjacent reverberation intervals if they are closer than the minimum interval
          between bursts.
        - The 'min_interval_between' parameter from 'bursts_params' is used to determine the minimum interval between bursts.
        """

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


    def _predict_reverbs(self, model_input, spikes_binary):
        """
        Predict reverberations using a pre-trained machine learning model.

        This method predicts reverberations based on the given model input and binary spike data.

        Parameters
        ----------
        model_input : array_like
            Input data for the model.
        spikes_binary : array_like
            Binary spike data.

        Returns
        -------
        list
            List of timestamp ranges corresponding to predicted reverberations.

        Notes
        -----
        - The model input can be either signal or spike data, specified by the 'input_type' parameter in 'model_params' attribute.
        - The method iterates through the input data in windows, predicts reverberations for each window, and combines the predictions.
        - The 'max_interval_start', 'max_interval_end', and 'min_interval_between' parameters for reverberations detection are predicted by the model.
        - Predicted reverberations are filtered based on the minimum duration criteria defined in 'reverbs_params' attribute.
        """

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
        return reverbs_pred


    def normalize_signal(self, signal):
        """
        Normalize the input signal.

        Parameters
        ----------
        signal : array_like
            Input signal to be normalized.

        Returns
        -------
        array_like
            Normalized signal.

        Notes
        -----
        - The normalization is performed by dividing the signal by its maximum absolute value.
        """
        return signal/abs(signal).max()
        
    def normalize_threshold(self, signal, threshold):
        """
        Normalize the threshold relative to the input signal.

        Parameters
        ----------
        signal : array_like
            Input signal used for normalization.
        threshold : float
            Threshold value to be normalized.

        Returns
        -------
        float
            Normalized threshold.

        Notes
        -----
        - The threshold is normalized by dividing it by the maximum absolute value of the input signal.
        """
        return threshold/abs(signal).max()

    def reduce_dimension(self, X, input_type = None, reduction_factor = None): 
        """
        Reduce the dimensionality of the input data.

        Parameters
        ----------
        X : array_like
            Input data to be dimensionally reduced.
        input_type : str, optional
            Type of input data ('signal' or 'spikes').
        reduction_factor : int, optional
            Factor by which to reduce the dimensionality.

        Returns
        -------
        array_like
            Dimensionally reduced input data.

        Notes
        -----
        - If 'input_type' is 'spikes', the dimensionality is reduced by selecting every 'reduction_factor' element.
        - If 'input_type' is 'signal', the dimensionality is reduced by averaging every 'reduction_factor' elements.
        """  

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
        """
        Reduce the dimensionality of the normalized absolute signal.

        Parameters
        ----------
        signal : array_like
            Input signal to be normalized and dimensionally reduced.

        Returns
        -------
        array_like
            Dimensionally reduced normalized absolute signal.
        """

        if signal.ndim == 1:
            return self.reduce_dimension(self.normalize_signal(abs(signal)), input_type = 'signal')
        else: 
            return np.array(list(map(self.reduce_dimension, self.normalize_signal(abs(signal)), ['signal' for _ in range(len(signal))])))



    def detect_net(self, inp, minChannelsParticipating=8, minSimultaneousChannels=6):
        """
        Detect net bursts or net reverbs.

        Parameters
        ----------
        inp : str
            Input type for which to detect the net ('reverbs' or 'bursts').
        minChannelsParticipating : int, optional
            Minimum number of channels participating to start a net burst.
        minSimultaneousChannels : int, optional
            Minimum number of simultaneous channels to continue a net burst.

        Notes
        -----
        - A net burst or net reverb is defined by bursts or reverbs occurring simultaneously on multiple channels.
        - The function detects net bursts or net reverbs based on the input type.
        - It starts a net burst when the number of channels bursting together exceeds 'minChannelsParticipating'.
        - It ends a net burst when the number of simultaneous channels drops below 'minSimultaneousChannels'.
        - The resulting net bursts or net reverbs are stored in the corresponding attributes ('net_reverbs' or 'net_bursts') 
          and their binary representations are stored in 'net_reverbs_binary' or 'net_bursts_binary', respectively.
        """
        
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
        """
        Save spikes data to a CSV file.

        Notes
        -----
        - The method saves spikes data to a CSV file containing columns for channel ID, channel label, well ID, well label, compound ID, compound name,
          experiment, dose in pM, dose label, and timestamp.
        - The CSV file is saved in the output folder with a filename based on the output name attribute and the type of data being saved 
          (e.g., '_REVERBS_PREDICTED.csv').
        """

        columns_spikes = ['Channel ID', 'Channel Label', 'Well ID', 'Well Label', 'Compound ID', 'Compound Name']
        columns_spikes.extend(['Experiment', 'Dose [pM]', 'Dose Label'])
        columns_spikes.extend(['timestamp'])
        spikes_data_file = os.path.join(self.output_folder, f'{self.output_name}_REVERBS_PREDICTED.csv')

        

    def analyze_dataset(self, file=None, mode='csv', save_default=False):
        """
        Analyze the datasets from a CSV file,  and save the results to CSV files.

        This method conducts a thorough analysis of the dataset, encompassing the detection of reverberations (reverbs),
        bursts, network bursts, and various statistics. It then saves the results to CSV files based on the specified mode
        and whether to save default analysis results.

        Parameters
        ----------
        file : str or None, optional
            The file name or path of the CSV with datasets to analyze. If None, the dataset associated with the 
            dataset attribute will be analyzed.
        mode : str, optional
            The mode of analysis. Default is 'csv', meaning it will receiva a CSV file with the dataset information.
        save_default : bool, optional
            Determines whether to also save the results of default analysis alongside model-based analysis. Default is False.

        Notes
        -----
        - The method first creates dataframes to store information about reverbs, bursts, network bursts, and statistics.
        - It loads the CSV file, iterates over each dataset, and analyzes each well speficified within the dataset.
        - For each well, it performs spike detection, threshold detection, and then applies the specified methods
          (default or model-based) for reverbs, bursts, and network bursts detection.
        - It calculates various statistics based on the detected bursts and network bursts.
        - Finally, it saves the results to separate CSV files according to the specified mode and whether to save default
          analysis results.
        """

        if self.output_folder != '':
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

    def plot_window(self, signal, start_time=None, duration=None, threshold=None, spikes=None, reverberations=None,
                    bursts=None, net_bursts=None, save=False, output_name=None, figsize=(6, 6), yunits='a.u.',
                    xunits='s'):
        """
        Plot a window of the signal with detected spikes, reverberations, bursts, and network bursts.

        Parameters
        ----------
        signal : array_like
            The input signal data.
        start_time : float, optional
            The start time of the window in seconds. Default is None (start from the beginning).
        duration : float, optional
            The duration of the window in seconds. Default is None (plot until the end of the signal).
        threshold : float, optional
            The threshold value for plotting. Default is None (no threshold line).
        spikes : array_like, optional
            The timestamps of detected spikes. Default is None (no spikes in the plot)
        reverberations : array_like, optional
            The timestamps data of detected reverberations. Default is None 
        bursts : array_like, optional
            The timestamps data detected bursts. Default is None
        net_bursts : array_like, optional
            The timestamps data of detected network bursts. Default is None
        save : bool, optional
            Whether to save the plot as a JPG file. Default is False.
        output_name : str, optional
            The name of the output JPG file if save is True. Default is None.
        figsize : tuple, optional
            The size of the figure (width, height) in inches. Default is (6, 6).
        yunits : str, optional
            The units of the y-axis. Default is 'a.u.' (arbitrary units).
        xunits : str, optional
            The units of the x-axis. Default is 's' (seconds).

        Notes
        -----
        - It normalizes the signal if yunits is 'a.u.' and converts it to millivolts (mV) if yunits is 'mV'.
        - Detected spikes, reverberations, bursts, and network bursts are overlaid on the plot with different colors.
        """

        if signal.ndim > 1:
            print("More than one signal (channel) input!")
            return

        if start_time is None: start_time = 0
        start_timestamp = int(start_time*self.samplingFreq)
        if duration is None: duration = int(len(signal)/self.samplingFreq)

        
        if start_timestamp > len(signal):
            print("Start time (in seconds) bigger than recorded signal time.")
            return

        duration_ts = int(duration*self.samplingFreq)
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

        if yunits.lower() == 'a.u.':
            sig = self.normalize_signal(signal)
        elif yunits.lower() == 'mv':
            sig = self.convert_signal(signal, self.adZero, self.conversionFactor, self.exponent)

        fig = plt.figure(figsize=figsize)



        for spike in spikes_to_plot:
            plt.axvline(spike/self.samplingFreq,0,0.022, c = '#660033', lw = 1)

        if yunits.lower() == 'a.u.':
            position_reverb = -1.24
            position_bursts = -1.164
            position_net_bursts = -1.07
        elif yunits.lower() == 'mv':
            position_reverb = 1.24*(min(sig))
            position_bursts = 1.164*(min(sig))
            position_net_bursts = 1.07*(min(sig))

        for reverb in reverbs_to_plot:
            plt.hlines(position_reverb,reverb[0]/self.samplingFreq,reverb[-1]/self.samplingFreq, colors = 'blue', lw = 6.5)

        for burst in bursts_to_plot:
            plt.hlines(position_bursts,burst[0]/self.samplingFreq,burst[-1]/self.samplingFreq, colors = '#FF3333', lw = 6.5)

        for net_bursts in net_bursts_to_plot:
            plt.hlines(position_net_bursts,net_bursts[0]/self.samplingFreq,net_bursts[-1]/self.samplingFreq, colors = 'orange', lw = 6.5)

        plt.plot(np.arange(start_timestamp, end_timestamp)/self.samplingFreq, sig[start_timestamp:end_timestamp], c = 'k', lw = 1)
        # plt.axhline(0, c = 'k', lw = 1)
        if threshold is not None:
            if yunits.lower() == 'a.u.':
                thresh = self.normalize_threshold(signal, threshold)
            elif yunits.lower() == 'mv':
                thresh = self.convert_threshold(threshold, self.adZero, self.conversionFactor, self.exponent)
            plt.axhline(thresh, color = 'k', lw = 2)
            plt.axhline(-thresh, color = 'k', lw = 2)
        plt.xlabel(f'Time [{xunits}]')
        plt.ylabel(f'Signal [{yunits}]')

        if yunits.lower() == 'a.u.':
            plt.ylim(-1.35,1.35)
        elif yunits.lower() == 'mv':
            plt.ylim(1.35*min(sig), 1.35*max(sig))
        plt.xlim(start_timestamp/self.samplingFreq, end_timestamp/self.samplingFreq)
        plt.grid(ls = 'dotted')
        if save:
            plt.savefig(f'{output_name}.jpg')
        plt.show()
        plt.close()


    def plot_raster(self, spikes, reverbs = None, bursts = None, net_reverbs = None, net_bursts = None):
        """
        Plot a raster plot of spikes with optional overlay of reverberations, bursts, and network bursts.

        Parameters
        ----------
        spikes : array_like
            The timestamps of spikes. Can be a list of timestamps for multiple channels.
        reverbs : array_like, optional
            The timestamps data of detected reverberations.
        bursts : array_like, optional
            The timestamps data of detected bursts.
        net_reverbs : array_like, optional
            The timestamps data of detected network reverberations.
        net_bursts : array_like, optional
            The timestamps data of detected network bursts.

        Returns
        -------
        None

        Notes
        -----
        - Each spike is represented by a vertical line at its timestamp.
        - Detected events are filled between their start and end timestamps.
        """
        
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
        """
        Plot a raster plot for a specific well with optional overlay of events.

        Parameters
        ----------
        file : str
            The filename or path of the data file.
        well : str
            The label or ID of the well to plot.
        method : str, optional
            The method used for detecting events. Default is 'default'.
        reverbs : bool, optional
            Whether to overlay detected reverberations on the raster plot. Default is False.
        bursts : bool, optional
            Whether to overlay detected bursts on the raster plot. Default is False.
        net_reverbs : bool, optional
            Whether to overlay detected network reverberations on the raster plot. Default is False.
        net_bursts : bool, optional
            Whether to overlay detected network bursts on the raster plot. Default is False.

        Returns
        -------
        None
        """

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
        


        

