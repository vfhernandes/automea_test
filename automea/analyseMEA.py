import numpy as np

## returns total number of spikes for a full well
def total_number_of_binary(spikes_binary):
    return np.where(spikes_binary.flatten() == 1)[0].shape[0]

## returns mean firing rate (spikes) in Hz for a full well
def mean_firing_rate(spikes_binary, total_seconds = 600):
    return np.round(np.where(spikes_binary == 1)[0].shape[0]/(total_seconds*spikes_binary.shape[0]), 2)

## returns total number of bursts for a full well 
def total_number_of_bursts(bursts_timestamp):
    number_of_bursts = 0
    for bursts_channel in bursts_timestamp:
        number_of_bursts += len(bursts_channel)
    return number_of_bursts

## returns mean bursting rate in ms for a full well
def mean_bursting_rate(bursts_timestamp, total_minutes = 10):
    number_of_bursts = 0
    non_zero_channels = 0
    for bursts_channel in bursts_timestamp:
        number_of_bursts += len(bursts_channel)
        if len(bursts_channel):
            non_zero_channels += 1
    if non_zero_channels:
        return np.round(number_of_bursts/total_minutes/non_zero_channels,2)
    else:
        return np.nan

## returns mean burst duration in ms for a full well
def mean_burst_duration(bursts_timestamp, input_unit = 'timestamp'):
    duration = 0
    number_of_bursts = 0
    for bursts_channel in bursts_timestamp:
        if len(bursts_channel):
            duration += sum(np.array(bursts_channel)[:,-1] - np.array(bursts_channel)[:,-0])
        number_of_bursts += len(bursts_channel)
    if number_of_bursts:
        if input_unit == 'timestamp':
            return np.round(duration/number_of_bursts/10,2)
        elif input_unit == 's':
            return np.round(1_000*duration/number_of_bursts,2)
    else:
        return np.nan

## returns mean inBurst frequency in Hz (number of spikes inside burst) for full well
def mean_inburst_frequency(spikes_binary_well, bursts_timestamp, samplingFreq=10_000):
    number_of_spikes_inside_bursts = 0
    number_of_bursts = 0
    for channel, spikes_binary in enumerate(spikes_binary_well):
        number_of_bursts += len(bursts_timestamp[channel])
        whereSpikes1 = np.where(spikes_binary == 1)[0]
        for i, burst in enumerate(bursts_timestamp[channel]):
            burst_duration_in_seconds = (burst[-1]-burst[0])/samplingFreq
            number_of_spikes_inside_bursts += len(whereSpikes1[(whereSpikes1 >= burst[0]) & (whereSpikes1 <= burst[-1])])/burst_duration_in_seconds
    if number_of_bursts:
        return np.round(number_of_spikes_inside_bursts/number_of_bursts,2)
    else:
        return np.nan

## returns interBurstTrain interval (interval between burst trains) in ms for full well
def mean_interBurstTrain_interval(merged_bursts, input_unit = 'timestamp'):
    intervals = []
    for channel, burstsChannel in enumerate(merged_bursts):
        if len(burstsChannel):
            burstsChannel = np.array(burstsChannel)
            burstsStart = burstsChannel[1:,0]
            burstsEnd = burstsChannel[:-1,1]
            intervals.extend(list((burstsStart - burstsEnd)))
    if len(intervals):
        mean = np.mean(intervals)    
        if input_unit == 'timestamp':
            return np.round(mean/10, 2)
        elif input_unit == 's':
            return np.round(mean*1_000, 2)
    else:
        return np.nan

## returns coefficience of variance for interBurstTrain interval in ms for full well
## (standard deviation over mean)
def coeff_variance_interBurstTrain_interval(merged_bursts, input_unit = 'timestamp'):
    intervals = []
    for channel, burstsChannel in enumerate(merged_bursts):
        if len(burstsChannel):
            burstsChannel = np.array(burstsChannel)
            burstsStart = burstsChannel[1:,0]
            burstsEnd = burstsChannel[:-1,1]
            intervals.extend(list((burstsStart - burstsEnd))) 
    if len(intervals):
        mean = np.mean(intervals, dtype=np.float64)
        coeff = np.std(intervals)/mean
        if input_unit == 'timestamp':
            return np.round(coeff,2)   
        elif input_unit == 's':
            return np.round(coeff*1_000,2)
    else:
        return np.nan

## calculates median of bursts per burst-train. Then returns average of median for whole well
def mean_of_median_bursts_per_burstTrain(bursts, mergedBursts, input_unit = 'timestamp'):
    burstsInMergedWell = np.zeros(len(mergedBursts))
    for channel, merged in enumerate(mergedBursts):
        if len(merged) and len(bursts[channel]):
            merged = np.array(merged)
            startsMerged, endsMerged = merged.T
            burstsChannel = np.array(bursts[channel])
            startsBursts, endsBursts = burstsChannel.T
            burstsInMergedChannel = []
            for merg in merged:
                listOfMatches = list(np.where(merg[0] <= startsBursts)[0])
                listOfMatches.extend(list(np.where(merg[1] >= endsBursts)[0]))
                burstsInMergedChannel.append(abs(np.unique(listOfMatches).shape[0] - len(listOfMatches)))
            burstsInMergedWell[channel] = np.median(burstsInMergedChannel)
    return np.round(np.mean(burstsInMergedWell[burstsInMergedWell!=0]), 2)



def mean_bursts_per_burstTrain(bursts, mergedBursts, input_unit = 'timestamp', net = False):

    if net is True:
        burstsInMergedChannel = 0
        if len(bursts) and len(mergedBursts):
            merged = np.array(mergedBursts)
            burstsChannel = np.array(bursts)
            startsBursts, endsBursts = burstsChannel.T
            for merg in merged:
                listOfMatches = list(np.where(merg[0] <= startsBursts)[0])
                listOfMatches.extend(list(np.where(merg[1] >= endsBursts)[0]))
                burstsInMergedChannel += abs(np.unique(listOfMatches).shape[0] - len(listOfMatches))
            return np.round(burstsInMergedChannel/len(mergedBursts), 2)
        else:
            return 0
    else:
        burstsInMergedChannel = []
        for channel, merged in enumerate(mergedBursts):
            if len(merged) and len(bursts[channel]):
                merged = np.array(merged)
                startsMerged, endsMerged = merged.T
                burstsChannel = np.array(bursts[channel])
                startsBursts, endsBursts = burstsChannel.T
                for merg in merged:
                    listOfMatches = list(np.where(merg[0] <= startsBursts)[0])
                    listOfMatches.extend(list(np.where(merg[1] >= endsBursts)[0]))
                    burstsInMergedChannel.append(abs(np.unique(listOfMatches).shape[0] - len(listOfMatches)))
        return np.round(np.mean(burstsInMergedChannel), 2)

def median_bursts_per_burstTrain(bursts, mergedBursts, input_unit = 'timestamp', net = False):

    if net is True:
        burstsInMergedChannel = []
        if len(bursts) and len(mergedBursts):
            merged = np.array(mergedBursts)
            burstsChannel = np.array(bursts)
            startsBursts, endsBursts = burstsChannel.T
            for merg in merged:
                listOfMatches = list(np.where(merg[0] <= startsBursts)[0])
                listOfMatches.extend(list(np.where(merg[1] >= endsBursts)[0]))
                burstsInMergedChannel.append(abs(np.unique(listOfMatches).shape[0] - len(listOfMatches)))
            return np.round(np.median(burstsInMergedChannel), 2)
        else:
            return 0
        
    else:    
        burstsInMergedChannel = []
        for channel, merged in enumerate(mergedBursts):
            if len(merged) and len(bursts[channel]):
                merged = np.array(merged)
                startsMerged, endsMerged = merged.T
                burstsChannel = np.array(bursts[channel])
                startsBursts, endsBursts = burstsChannel.T
                for merg in merged:
                    listOfMatches = list(np.where(merg[0] <= startsBursts)[0])
                    listOfMatches.extend(list(np.where(merg[1] >= endsBursts)[0]))
                    burstsInMergedChannel.append(abs(np.unique(listOfMatches).shape[0] - len(listOfMatches)))
        return np.round(np.median(burstsInMergedChannel), 2)





## returns percentage of spikes that are not in net bursts - for whole channel
def random_spikes_percentage(spikes_binary, netBursts_timestamp):
    number_of_spikes_inside_net = 0
    number_of_spikes = 0
    for channel in range(len(spikes_binary)):
        whereSpikes1 = np.where(spikes_binary[channel] == 1)[0]        
        number_of_spikes += len(whereSpikes1)
        for _, net in enumerate(netBursts_timestamp[channel]):
            number_of_spikes_inside_net += len(whereSpikes1[(whereSpikes1 >= net[0]) & (whereSpikes1 <= net[-1])])
    random_spikes = number_of_spikes - number_of_spikes_inside_net
    return np.round(100*(random_spikes/number_of_spikes), 2)


def random_spikes_percentage_net(spikes_binary, netBursts_timestamp):
    number_of_spikes_inside_net = 0
    number_of_spikes = 0
    for channel in range(len(spikes_binary)):
        whereSpikes1 = np.where(spikes_binary[channel] == 1)[0]        
        number_of_spikes += len(whereSpikes1)
        for _, net in enumerate(netBursts_timestamp):
            number_of_spikes_inside_net += len(whereSpikes1[(whereSpikes1 >= net[0]) & (whereSpikes1 <= net[-1])])
        random_spikes = number_of_spikes - number_of_spikes_inside_net
    return np.round(100*(random_spikes/number_of_spikes), 2)

def random_spikes_percentage_old(spikes_binary, netBursts_timestamp):
    number_of_spikes_inside_net = 0
    number_of_spikes = 0
    for channel in range(len(spikes_binary)):
        whereSpikes1 = np.where(spikes_binary[channel] == 1)[0]        
        number_of_spikes += len(whereSpikes1)
        for _, net in enumerate(netBursts_timestamp):
            number_of_spikes_inside_net += len(whereSpikes1[(whereSpikes1 >= net[0]) & (whereSpikes1 <= net[-1])])
    random_spikes = number_of_spikes - number_of_spikes_inside_net
    return np.round(100*(random_spikes/number_of_spikes), 2)


## returns total number of bursts
def total_number_of_netBursts(netbursts_timestamp):
    return len(netbursts_timestamp)

## returns mean of net burst rate - net bursts / minute
def mean_netbursting_rate(netbursts_timestamp, total_minutes = 10):
    number_of_bursts = len(netbursts_timestamp)
    return np.round(number_of_bursts/total_minutes,2)    

## returns mean net burst duration in ms
def mean_netburst_duration(netbursts, input_unit = 'timestamp'):
    if len(netbursts):
        duration = sum(np.array(netbursts)[:,-1] - np.array(netbursts)[:,-0])
        if input_unit == 'timestamp':
            return np.round(duration/len(netbursts)/10,2)
        elif input_unit == 's':
            return np.round(1_000*duration/len(netbursts),2)
    else:
        return np.nan

## return mean in-net burst frequency in Hz
## spiking frequency inside net bursts - for whole well
def mean_innetburst_frequency(spikes_binary_well, netbursts_timestamp,samplingFreq=10_000):
    number_of_spikes_inside_bursts = 0
    number_of_bursts = len(netbursts_timestamp)
    if number_of_bursts:
        for channel, spikes_binary in enumerate(spikes_binary_well):
            whereSpikes1 = np.where(spikes_binary == 1)[0]
            for i, burst in enumerate(netbursts_timestamp):
                burst_duration_in_seconds = (burst[-1]-burst[0])/samplingFreq
                number_of_spikes_inside_bursts += len(whereSpikes1[(whereSpikes1 >= burst[0]) & (whereSpikes1 <= burst[-1])])/burst_duration_in_seconds
        return np.round(number_of_spikes_inside_bursts/(len(spikes_binary_well)*number_of_bursts),2)
    else:
        return np.nan

## return mean inter net burst-train interval
## distance between each net burst train
def mean_interNetBurstTrain_interval(merged_bursts, input_unit = 'timestamp'):
    intervals = []
    if len(merged_bursts) > 1:
        merged_bursts = np.array(merged_bursts)
        burstsStart = merged_bursts[1:,0]
        burstsEnd = merged_bursts[:-1,1]
        intervals = burstsStart - burstsEnd
        mean = np.mean(intervals)
        if input_unit == 'timestamp':
            return np.round(mean/10, 2)
        elif input_unit == 's':
            return np.round(mean*1_000, 2)
    else:
        return np.nan


## returns coefficient of variance for inter net burst-train interval
## std deviation / mean of intervals between net burst-trains
def coeff_variance_interNetBurstTrain_interval(merged_bursts, input_unit = 'timestamp'):
    merged_bursts = np.array(merged_bursts)
    if len(merged_bursts) > 1:
        burstsStart = merged_bursts[1:,0]
        burstsEnd = merged_bursts[:-1,1]
        intervals = burstsStart - burstsEnd
        mean = np.mean(intervals, dtype=np.float64)
        coeff = np.std(intervals)/mean

        if input_unit == 'timestamp':
            return np.round(coeff,2)   
        elif input_unit == 's':
            return np.round(coeff*1_000,2)
        
    else:
        return np.nan

def number_of_spikes_inside_burst(spikes_binary, burst):
    number_inside = 0
    for channel in range(len(spikes_binary)):
        where1 = np.where(spikes_binary[channel] == 1)[0]
        number_inside += len(where1[(where1 >= burst[0]) & (where1 <= burst[-1])])
    return number_inside
