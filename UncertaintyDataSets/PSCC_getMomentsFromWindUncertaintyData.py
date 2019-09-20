''' Data Processing to calculate the moments from empirical data of wind power forecast errors observed in Denmark from  2006-2007 '''
''' For details refer to: 
        Pinson, Pierre. Wind Energy: Forecasting Challenges for Its Operational Management. 
        Statist. Sci. 28 (2013), no. 4, 564--585. doi:10.1214/13-STS445. https://projecteuclid.org/euclid.ss/1386078879
        PDF Link: http://orbit.dtu.dk/files/61111946/pinson13_windstat.pdf
'''

'''The available 10,000 scenarios are split into two sets - Train (500) and Test (9500). Train set is used for computing the moments uesd by 
the chance constraint reformulation. ''' 

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt


#Step 1: Gather data from excel files into dataframes for each zone and concatenate them to form a large data frame indexed by ZoneNum_HourNum  and with columns
# PointForecast and Scen1 through Scen10000.
ForecastErrors = pd.DataFrame()
TestData = pd.DataFrame()
TrainData = pd.DataFrame()
FileNameBase_Scenarios = "wp-scen-zone"

# !!!!! MUST BE CHANGED WITH CHANGE IN NETWORK AND SIMULATION PARAMETERS!!!!
NumZones = 15   #Total Number of Zones for which data is available - Constant!
NumHours = 1

#Number of WindFarms and Location
NumWindFarms = 2
WindFarmData = {
    'WindFarm':{
        'k1':{'Wmax':500, 'BusNum': 'n5', 'ZoneNum': 1},
        'k2':{'Wmax':500, 'BusNum': 'n7', 'ZoneNum': 2},
    }
}

NumTrainingSamples = 1000

for w in range(1, NumWindFarms+1):
    #Get Zone data for this Wind Farm
    i = WindFarmData['WindFarm']['k'+str(w)]['ZoneNum']
    FileName = FileNameBase_Scenarios + str(i)+ ".xlsx"
    WFZoneScenariosData = pd.read_excel("scenarioset/"+FileName, header=None).head(NumHours).transpose()    #Selecting the first 24 hour values of the dataset
    #Obtain the Zone Capacity Multiplier
    WFZoneScenariosData = WFZoneScenariosData.select_dtypes(exclude=['object', 'datetime'])*WindFarmData['WindFarm']['k'+str(w)]['Wmax']
    WFZoneScenariosData_sampled = WFZoneScenariosData.head(NumTrainingSamples)     # First n rows used for computing moments
    WFZoneHourlyWindPointForecast = pd.DataFrame(WFZoneScenariosData_sampled.mean(axis=0)).transpose()              #TBC: Estimating the measured value based on mean of all point forecasts
    WFZoneErrorsDataWithPF = WFZoneHourlyWindPointForecast.append(WFZoneScenariosData_sampled, ignore_index=True)
    WFZoneErrorsDataWithPF = WFZoneErrorsDataWithPF - WFZoneErrorsDataWithPF.iloc[0]
    WFZoneErrorsDataWithPF.update(WFZoneHourlyWindPointForecast)
    WFZoneErrorsDataWithPF.columns = ['H'+str(x) for x in np.arange(1, NumHours+1)]
    WFTempDataFrame = (pd.DataFrame(['WFk'+str(w)+'Z'+str(i)+'PointFor'])).append(['WFk'+str(w)+'Z'+str(i)+'Scen'+ str(x) for x in np.arange(1, 10001)], ignore_index=True)
    WFZoneErrorsDataWithPF['Identifier'] = WFTempDataFrame
    WFZoneErrorsDataWithPF.set_index('Identifier')
    ForecastErrors = ForecastErrors.append(WFZoneErrorsDataWithPF, ignore_index=False)
    '''
    #Using first n rows for the Train dataset
    WFZoneScenarios_TrainData = WFZoneScenariosData.head(NumTrainingSamples)
    WFZoneScenarios_TrainData.columns = ['H'+str(x) for x in np.arange(1, NumHours+1)]
    WFZoneScenarios_TrainData_IdentifierCol = pd.DataFrame(['WFk'+str(w)+'Z'+str(i)+'Scen'+ str(x) for x in np.arange(1, 10001)])
    WFZoneScenarios_TrainData['IdentifierCol'] = WFZoneScenarios_TrainData_IdentifierCol
    TrainData = TrainData.append(WFZoneScenarios_TrainData, ignore_index=True) 
    #Using remainder of rows data for the Test Dataset
    WFZoneScenarios_TestData = WFZoneScenariosData.tail(10000-NumTrainingSamples)
    WFZoneScenarios_TestData.columns = ['H'+str(x) for x in np.arange(1, NumHours+1)]
    WFZoneScenarios_TestData_IdentifierCol = pd.DataFrame(['WFk'+str(w)+'Z'+str(i)+'Scen'+ str(x) for x in np.arange(1, 10001)])
    WFZoneScenarios_TestData['IdentifierCol'] = WFZoneScenarios_TestData_IdentifierCol
    TestData = TestData.append(WFZoneScenarios_TestData, ignore_index=True) '''
print(ForecastErrors.head())


# Step 2: Calculating the large covariance matrix of dimensions (NumHours*NummWindFarms, NumHours*NumWindFarms)
SpatialTemporalCovariance = np.array([], dtype=np.int64).reshape(0, NumHours*NumWindFarms)
for hour_num in range(1, NumHours+1):
    first_hour = 'H'+str(hour_num)                  #For temporal dimension, the first hour
    TemporalCovariancesThisHourSubMatrices = []
    for hour_iterval in range(1, NumHours+1):       
        second_hour = 'H'+str(hour_iterval)         #For temporal dimension, iterating over the next hours
        #print('Creating Intertemporal Covariance Submatrices for Hours {} and {}'.format(first_hour, second_hour))
        ThisHourCovarianceValues = []
        for wf_num in range(1, NumWindFarms+1):
            first_wf = 'WFk'+str(wf_num)         #For spatial dimension, the first wind_farm
            TempVal = []
            for wf_iterval in range(1, NumWindFarms+1):
                second_wf = 'WFk'+str(wf_iterval) #For spatial dimension, the second zone
                #print('Covariance value between Wind Farm Forecast Errors : {} and {}. Hours are: {} and {}'.format(first_wf, second_wf, first_hour, second_hour))
                TempVal.append(np.asarray((ForecastErrors.loc[ForecastErrors['Identifier'].str.contains(first_wf), first_hour][1:]).cov((ForecastErrors.loc[ForecastErrors['Identifier'].str.contains(second_wf), second_hour][1:]))))
            ThisHourCovarianceValues = np.reshape(np.append(ThisHourCovarianceValues, TempVal), (-1, NumWindFarms))
            #print('Hourwise covariance values: \n {}'.format(ThisHourCovarianceValues))
        TemporalCovariancesThisHourSubMatrices = np.append(TemporalCovariancesThisHourSubMatrices, ThisHourCovarianceValues)
        #print(np.transpose(np.reshape(TemporalCovariancesThisHourSubMatrices, (-1, NumWindFarms))))
        RowsVal = np.transpose(np.reshape(TemporalCovariancesThisHourSubMatrices, (-1, NumWindFarms)))
        #print('RowsVal --- Shape: {}, \n {}'.format(RowsVal.shape, RowsVal))
    SpatialTemporalCovariance = np.vstack((SpatialTemporalCovariance, RowsVal))
print(SpatialTemporalCovariance)

'''
#Exporting to CSV through Pandas. Output Matrix Size : NumHours*NumWindFarms X NumHours*NumWindFarms
CovarMatrix = pd.DataFrame(SpatialTemporalCovariance)
#CovarMatrix.to_csv("2_WFs_Covariance_Matrix_Data.csv", index=False, header=None)


# ----------------- Temporal Folding of Covariance Values ---------------- #
#Step 3: Temporal Folding (summation) of covariance matrix for usage by the chance-constrained recourse policy decision
TemporallyFoldedCovarMatrix = np.zeros((NumHours, NumHours))
for i in range(NumHours):
    for j in range(NumHours):
        TemporallyFoldedCovarMatrix[i, j] = SpatialTemporalCovariance[NumWindFarms*i:NumWindFarms*(i+1), NumWindFarms*j:NumWindFarms*(j+1)].sum()
#print(TemporallyFoldedCovarMatrix)

#Exporting Temporally Folded CovarianceMatrix to CSV through Pandas. Output Matrix Size = NumHours X NumHours
TempFoldedCovarMatrix = pd.DataFrame(TemporallyFoldedCovarMatrix)
#TempFoldedCovarMatrix.to_csv("2_WFs_TemporallyFolded_Covariance_Matrix_Data.csv", index=False, header=None)


# --------------- Extracting Point Forecast Values ------------- #
#Step 5: Extracting the Point forecast values scaled for each zone's installation
PointForecasts = ForecastErrors.loc[ForecastErrors['Identifier'].str.contains('Point')]
#PointForecasts.to_csv("2_WFs_500MW_Point_Forecast_Values.csv", index=False, header=None)


# ---- Exporting Test Data ------ #
TestData.to_csv("2_WFs_500MW_TestData.csv", index = False, header=None)
TrainData.to_csv("2_WFs_500MW_TrainData.csv", index=False, header=None)
'''