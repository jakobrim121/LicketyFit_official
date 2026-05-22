import numpy as np
import matplotlib
import sys
sys.path.insert(0, "../event_display")
from EventDisplay import EventDisplay

sys.path.insert(0, "../LicketyFit")
sys.path.insert(0, "../tables")

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







# """
# Optional WCSim/LF event plotting helper.

# This module intentionally has no import-time side effects.  The original version
# constructed an EventDisplay object at import time, which could fail in a clean
# self-contained checkout when optional plotting assets were not present.
# """

# from __future__ import annotations

# from pathlib import Path

# import numpy as np


# def load_wcsim_to_wcte_mapping(mapping_path=None):
#     if mapping_path is None:
#         mapping_path = Path(__file__).resolve().parent.parent / "tables" / "wcsim_wcte_mapping.txt"
#     mapping_path = Path(mapping_path)
#     arr = np.loadtxt(mapping_path)
#     mapping = {}
#     for row in np.atleast_2d(arr):
#         # WCSim file convention in your previous plotting scripts used raw tube
#         # numbers offset by -1; the value returned is WCTE slot*100+pmt_0based.
#         mapping[int(row[0]) - 1] = int(row[1] * 100 + row[2] - 1)
#     return mapping


# def plot_event(file_path_or_data, evt_num, *, use_mapping=True, mapping_path=None):
#     """
#     Return an LF-style event array with columns [pmt_id, charge, time].

#     Parameters
#     ----------
#     file_path_or_data : str, Path, or dict-like
#         Either a WCSim npz path or a loaded dict/npz object with digi_hit_pmt,
#         digi_hit_charge, and digi_hit_time.
#     evt_num : int
#         Event index inside the WCSim arrays.
#     use_mapping : bool
#         If True, convert raw WCSim PMT IDs to WCTE PMT IDs using the mapping file.
#     """
#     if isinstance(file_path_or_data, (str, Path)):
#         data = np.load(file_path_or_data, allow_pickle=True)
#     else:
#         data = file_path_or_data

#     pmts = np.asarray(data["digi_hit_pmt"][evt_num], dtype=int)
#     charges = np.asarray(data["digi_hit_charge"][evt_num], dtype=float)
#     times = np.asarray(data["digi_hit_time"][evt_num], dtype=float)

#     if use_mapping:
#         mapping = load_wcsim_to_wcte_mapping(mapping_path)
#         mapped = []
#         keep = []
#         for i, p in enumerate(pmts):
#             if int(p) in mapping:
#                 mapped.append(mapping[int(p)])
#                 keep.append(i)
#         pmts = np.asarray(mapped, dtype=int)
#         charges = charges[keep]
#         times = times[keep]

#     return np.column_stack([pmts, charges, times])


# # def plot_event(*args, **kwargs):
# #     """
# #     Optional legacy display wrapper.

# #     The full 2D unwrapped event-display stack is not required for fitting and is
# #     not bundled here.  Use lf_array_from_wcsim_npz(...) to get the data array for
# #     fitting or copy your external event_display assets into the checkout if you
# #     need the old display.
# #     """
# #     raise RuntimeError(
# #         "The legacy 2D EventDisplay assets are optional and are not included in "
# #         "this self-contained fitter package. Use lf_array_from_wcsim_npz(...) "
# #         "to build fit arrays, or add your event_display package/assets and "
# #         "restore your plotting wrapper."
# #     )
