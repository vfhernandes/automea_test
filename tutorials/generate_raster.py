#!/usr/bin/env python
# coding: utf-8

# ## Generate images from datasets and files from previous analyses containing spikes and network bursts.

# In[1]:


import automea_copy
import numpy as np 
import pandas as pd 
import os  




# In[2]:


path_to_dataset = '/Volumes/T7/automea_files_to_plot/'

# folder_datasets = [('Fig3-4', 'MJ22C002-1823_DIV14.h5'),
#                    ('RHEB_DIV14', 'AH22C001-3039_DIV14.h5'),
#                    ('RHEB_DIV07', 'AH22C001-3039_DIV7-005.h5'),
#                    ('RHEB_DIV07', 'MJ22C001-1799_DIV7-004.h5'),
#                    ('RHEB_DIV07', 'MJ22C002-1782_DIV7.h5'),
#                    ('RHEB_DIV07', 'MJ22C003-1788_DIV7-002.h5'),
#                    ('CAMK2G_DIV18', '20180610_No_compound_m230518_hip2_DIV18_mwd.h5')
#                    ]
folder_datasets = [
                   ('RHEB_DIV14', 'MJ22C001-1799_DIV14-001.h5'),
                   ('RHEB_DIV14', 'MJ22C001-1799_DIV14-001.h5'),
                   ('RHEB_DIV14', 'MJ22C001-1799_DIV14-001.h5'),

                   ]

# well_timestamps_nets = [[('B5', 'Timestamps_MJ22C002_1823_B5.csv', 'NetworkBursts_MJ22C002_1823_B5.csv')],
#                         [('D6', 'Timestamps_AH22C001_3039_DIV14_D6.csv', 'NetworkBursts_AH22C001_3039_DIV14_D6.csv')],
#                         [('A6', 'Timestamps_AH22C001_3039_DIV07_A6.csv', 'NetworkBursts_AH22C001_3039_DIV07_A6.csv')],
#                         [('B3', 'Timestamps_MJ22C001_1799_DIV07_B3.csv', 'NetworkBursts_MJ22C001_1799_DIV07_B3.csv')],
#                         [('D5', 'Timestamps_MJ22C002_1782_DIV07_D5.csv', 'NetworkBursts_MJ22C002_1782_DIV07_D5.csv')],
#                         [('A2', 'Timestamps_MJ22C003_1788_DIV07_A2.csv', 'NetworkBursts_MJ22C003_1788_DIV07_A2.csv')],
#                         [('A1', 'Timestamps_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv', 'NetworkBursts_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv'),
#                          ('B1', 'Timestamps_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv', 'NetworkBursts_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv'),
#                          ('C2', 'Timestamps_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv', 'NetworkBursts_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv'),
#                          ('B3', 'Timestamps_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv', 'NetworkBursts_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv'),
#                          ('C1', 'Timestamps_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv', 'NetworkBursts_20180610_No_compound_m230518_hip2_DIV18_mwd.h5.csv'),
#                          ]]

well_timestamps_nets = [
                         [('D1', 'Timestamps_MJ22C001_1799_DIV14.csv', 'NetworkBursts_MJ22C001_1799_DIV14.csv'),
                         ('D3', 'Timestamps_MJ22C001_1799_DIV14.csv', 'NetworkBursts_MJ22C001_1799_DIV14.csv'),
                         ('B3', 'Timestamps_MJ22C001_1799_DIV14.csv', 'NetworkBursts_MJ22C001_1799_DIV14.csv')
                         ]]


am = automea_copy.Analysis()

am.path_to_dataset = path_to_dataset
am.loadh5(folder_datasets[0][-1])
signal_ = np.copy(am.signal)
am.model_name = 'signal30.h5'
am.loadmodel()



for folder_index, fd in enumerate(folder_datasets):
    for wtn in well_timestamps_nets[folder_index]:

        folder = fd[0]
        dataset = fd[1]
        well = wtn[0]
        spikes_manual_file = wtn[1]
        netbursts_manual_file = wtn[2]

        print(folder)
        print(dataset)
        print(well)
        print(spikes_manual_file)
        print(netbursts_manual_file, '\n\n')

        #am = automea_copy.Analysis()


        # In[3]:

        plots_dir = f'plots3/{dataset}_{well}'
        if os.path.exists(plots_dir) is False:
            os.mkdir(plots_dir)
            os.mkdir(plots_dir+'/signals')
            os.mkdir(plots_dir+'/rasters')



        # In[4]:


        # am.path_to_dataset = path_to_dataset
        # am.loadh5(dataset)


        # # In[5]:



        # In[6]:


        #signal_ = np.copy(am.signal)


        # In[7]:


        am.loadsignal(signal_[am.wellsFromData == am.wellLabelIndexDict[well]])


        # In[8]:


        # am.model_name = 'signal30.h5'
        # am.loadmodel()


        # In[9]:


        am.detect_threshold()
        am.detect_spikes()
        am.detect_reverbs(method = 'model')
        am.detect_bursts()
        am.detect_net('reverbs')
        am.detect_net('bursts')
        am.threshold, am.spikes, am.net_bursts


        # In[10]:


        path_to_manual_files = f'/Volumes/T7/automea_files_to_plot/Examples list/{folder}/'
        #spikes_manual_file = 'Timestamps_MJ22C002_1782_DIV07_D5.csv'
        #netbursts_manual_file = 'NetworkBursts_MJ22C002_1782_DIV07_D5.csv'


        # In[11]:

        #if fd != folder_datasets[-1]:
        #    spikes_manual_df = pd.read_csv(path_to_manual_files + spikes_manual_file, encoding='unicode_escape')
        #else:
        #    spikes_manual_df = pd.read_csv(path_to_manual_files + spikes_manual_file, encoding='unicode_escape', sep = '\t')
        spikes_manual_df = pd.read_csv(path_to_manual_files + spikes_manual_file, encoding='unicode_escape')




        # In[13]:


        channel_labels = np.unique(spikes_manual_df[spikes_manual_df['Well Label'] == well]['ï»¿"Channel ID"'].iloc[:].values)


        # In[14]:


        spikes_manual = []
        for channel_label in channel_labels:
            spikes_manual.append([])
            for index in spikes_manual_df[(spikes_manual_df['ï»¿"Channel ID"'] == channel_label) & (spikes_manual_df['Well Label'] == well)].index:
                spikes_manual[-1].append(spikes_manual_df['Timestamp [Âµs]'][index]//100)


        # In[15]:


        netbursts_manual_df = pd.read_csv(path_to_manual_files + netbursts_manual_file, encoding='unicode_escape')


        # In[16]:


        net_manual = []
        for index in netbursts_manual_df[netbursts_manual_df['Well Label']==well].index:
            start_ts = netbursts_manual_df['Start timestamp [Âµs]'][index]
            duration = netbursts_manual_df['Duration [Âµs]'][index]
            net_manual.append([start_ts//100, (start_ts+duration)//100])


        # In[17]:


        duration = 60
        duration_shorter = 5

        for i, start_time in enumerate(np.random.randint(0,500,size=3)):

            end_time = start_time + duration

            am.plot_raster(am.spikes, net_bursts = am.net_reverbs, start_time = start_time, end_time = end_time, 
                        save = f'{plots_dir}/rasters/raster_start{start_time}_duration{duration}_allwell_prednetreverbs.eps', show = False)
            am.plot_raster(am.spikes, net_bursts = net_manual, start_time = start_time, end_time = end_time, 
                        save = f'{plots_dir}/rasters/raster_start{start_time}_duration{duration}_allwell_manual.eps', show = False)

            for j in range(3):
                start_time_shorter = start_time + np.random.randint(0,95)
                channel_random  = np.random.randint(len(am.signal))
                end_time_shorter = start_time_shorter + duration_shorter
                am.plot_raster(am.spikes[channel_random], net_bursts = am.net_reverbs, start_time = start_time_shorter, end_time = end_time_shorter, 
                                save = f'{plots_dir}/rasters/raster_start{start_time}_duration{duration}_zoomedstart{start_time+start_time_shorter}_zoomedduration{duration_shorter}_prednetreverbs.eps', show = False)
                am.plot_raster(am.spikes[channel_random], net_bursts = net_manual, start_time = start_time_shorter, end_time = end_time_shorter,
                                save = f'{plots_dir}/rasters/raster_start{start_time}_duration{duration}_zoomedstart{start_time+start_time_shorter}_zoomedduration{duration_shorter}_manual.eps', show = False)


        # In[19]:


        duration_shorter = 5

        for _ in range(20):
            random_net = am.net_reverbs[np.random.randint(len(am.net_reverbs))]
            middle_of_net = random_net[0] + ((random_net[-1] - random_net[0])/2)
            start_time = middle_of_net/am.samplingFreq - 2.5
            channel_random  = np.random.randint(len(am.signal))


            am.plot_window(am.signal[channel_random], start_time=start_time, duration=duration_shorter,
                            threshold=am.threshold[channel_random], spikes = am.spikes[channel_random],
                            bursts=am.net_reverbs, net_bursts=net_manual, yunits = 'V', show = False,
                            save = f'{plots_dir}/signals/signal_channel{channel_random}_start{start_time}_duration5.eps')




