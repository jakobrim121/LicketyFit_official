import numpy as np
import matplotlib
import sys
sys.path.insert(0, "../event_display")
from EventDisplay import EventDisplay

sys.path.insert(0, "../LicketyFit")

#sys.path.insert(0, "../../")
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.colors as colors

#from Geometry.Device import Device
#from Emitter import Emitter

#from LicketyFit.Emitter import Emitter
# from LicketyFit.Event import SimulatedEvent
# from LicketyFit.PMT import PMT
# from LicketyFit.Fitter import Fitter

#1) make an instance of the event display class
eventDisplay = EventDisplay() 

#2) start by loading in the CSV file for how the mPMTs are mapped to 2d event display
#unwraps based on the mPMT slot ID 
eventDisplay.load_mPMT_positions('mPMT_2D_projection_angles.csv')

#mask out mPMT slots - newer WCSim doesn't have these mPMTs loaded  
#WCTE slot numbering
# eventDisplay.mask_mPMTs([45,77,79,27,32,85,91,99,12,14,16,18])
#WCSim container numbering
# eventDisplay.mask_mPMTs([20,73,38,49,55,65,67,33,71,92,101,95])

#3) load the WCSim mapping tube no to slot number
#for WCSim using the numpy output we need the mapping between the tube_number in WCSim and the slot and mPMT number in the detector
#this can be obtained from the geofile that WCSim produces 
#This changes if the CDS is implemented or not
eventDisplay.load_wcsim_tubeno_mapping("geofile_WCTE.txt")

wcte_mapping = np.loadtxt('../tables/wcsim_wcte_mapping.txt')

# wcsim uses positions 1-19, so have to subtract 1 in the mapping...
sim_wcte_mapping = {}
for i in range(len(wcte_mapping)):
    sim_wcte_mapping[int(wcte_mapping[i][0])-1] = int(wcte_mapping[i][1]*100 + wcte_mapping[i][2] - 1)


def plot_event(file_path_or_data, evt_num, vmax=5,vmin=0.1, log_scale=False, LF_data = True):
    
    if LF_data:
        data = {'digi_hit_time':[],'digi_hit_pmt':[],'digi_hit_charge':[]}
        try:
            simulated_event = np.load(file_path_or_data)

            

            # for mpmt in range(106):
            #     for pmt in range(19):
            #         try:
            # #             data['digi_hit_time'].append(simulated_event.expected_hit_times[mpmt][pmt][0])
            # #             data['digi_hit_pmt'].append(int(100*mpmt+pmt))
            # #             data['digi_hit_charge'].append(simulated_event.expected_hit_pe[mpmt][pmt][0])


            #         except:
            #             continue

            for i in range(len(simulated_event)):
                data['digi_hit_pmt'].append(simulated_event[i,0])
                data['digi_hit_charge'].append(simulated_event[i,1])

            #load the data to plot

            eventID =evt_num

            #mask = data["digi_hit_time"][eventID] < 20

            tube_no = data["digi_hit_pmt"]

            #data_to_plot = np.array(data["digi_hit_charge"])*8/max(data["digi_hit_charge"])
            data_to_plot = np.array(data["digi_hit_charge"])

            pmt_lf = np.asarray(data["digi_hit_pmt"])
            charge_lf = np.asarray(data["digi_hit_charge"])

            mPMT_id = []
            PMT_pos = []

            for i in range(len(pmt_lf)):
                mPMT_id.append(int(pmt_lf[i]/100))
                PMT_pos.append(int(pmt_lf[i]%100))

            #map from the tube number to the mPMT slot and position number
            #mPMT_id, PMT_pos = eventDisplay.map_wcsim_tubeno_to_slot_pmt_id(tube_no)
            data_to_plot = eventDisplay.process_data(mPMT_id,PMT_pos,data_to_plot)
            x = eventDisplay.plotEventDisplay(data_to_plot,vmax=vmax,vmin=vmin,log_scale=log_scale,color_norm=colors.Normalize(), style= "dark_background")
            
        except:
            
            for i in range(len(file_path_or_data)):
                data['digi_hit_pmt'].append(file_path_or_data[i,0])
                data['digi_hit_charge'].append(file_path_or_data[i,1])

            
            #load the data to plot

            eventID =evt_num

            #mask = data["digi_hit_time"][eventID] < 20

            tube_no = data["digi_hit_pmt"]

            #data_to_plot = np.array(data["digi_hit_charge"])*8/max(data["digi_hit_charge"])
            data_to_plot = np.array(data["digi_hit_charge"])

            pmt_lf = np.asarray(data["digi_hit_pmt"])
            charge_lf = np.asarray(data["digi_hit_charge"])

            mPMT_id = []
            PMT_pos = []

            for i in range(len(pmt_lf)):
                mPMT_id.append(int(pmt_lf[i]/100))
                PMT_pos.append(int(pmt_lf[i]%100))

            #map from the tube number to the mPMT slot and position number
            #mPMT_id, PMT_pos = eventDisplay.map_wcsim_tubeno_to_slot_pmt_id(tube_no)
            data_to_plot = eventDisplay.process_data(mPMT_id,PMT_pos,data_to_plot)
            x = eventDisplay.plotEventDisplay(data_to_plot,vmax=vmax,vmin=vmin,log_scale=log_scale,color_norm=colors.Normalize(), style= "dark_background")
            
        
    else:
        
        data2 = np.load(file_path_or_data, allow_pickle=True)

        eventID =evt_num

        true_digi = 'digi'

        mask = data2[true_digi+"_hit_time"][eventID] < 17

        tube_no = data2[true_digi+"_hit_pmt"][eventID][mask]+1

        data_to_plot = data2["digi_hit_charge"][eventID][mask]


        pmt = np.asarray(data2[true_digi+"_hit_pmt"][eventID][mask])
        charge = np.asarray(data2["digi_hit_charge"][eventID][mask])



        #map from the tube number to the mPMT slot and position number
        mPMT_id, PMT_pos = eventDisplay.map_wcsim_tubeno_to_slot_pmt_id(tube_no)
        data_to_plot = eventDisplay.process_data(mPMT_id,PMT_pos,data_to_plot)
        x = eventDisplay.plotEventDisplay(data_to_plot,vmax=vmax,vmin=vmin,log_scale=False,color_norm=colors.Normalize(), style= "dark_background")


