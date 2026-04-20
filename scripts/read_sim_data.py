import numpy as np
import matplotlib.pyplot as plt


def read_sim_data(file_path):
    
    sim_g = np.load(file_path,allow_pickle=True)
    
    true_hit_pmt = sim_g['true_hit_pmt']
    true_hit_time = sim_g['true_hit_time']

    digi_hit_pmt = sim_g['digi_hit_pmt']
    digi_hit_time = sim_g['digi_hit_time']

    track_start_position = sim_g['track_start_position']
    track_id = sim_g['track_id']
    track_pid = sim_g['track_pid']
    
    track_start_time = sim_g['track_start_time']
    
    position = sim_g['position']
    direction = sim_g['direction']
    energy = sim_g['energy']
    track_energy = sim_g['track_energy']
    track_stop_position = sim_g['track_stop_position']
    track_boundary_kes = sim_g['track_boundary_kes']
    
    track_parent = sim_g['track_parent']

    digi_hit_charge = sim_g['digi_hit_charge']
    
    sim_dict = {'true_hit_pmt':true_hit_pmt,'true_hit_time':true_hit_time,'digi_hit_pmt':digi_hit_pmt,
                'digi_hit_time':digi_hit_time, 'track_start_position':track_start_position, 'track_stop_position':track_stop_position, 'track_id':track_id,
               'track_pid':track_pid, 'track_start_time':track_start_time, 'track_parent':track_parent, 'digi_hit_charge':digi_hit_charge, 'position':position, 'direction':direction, 'energy':energy, 'track_energy':track_energy, 'track_boundary_kes':track_boundary_kes}
    
    return sim_dict