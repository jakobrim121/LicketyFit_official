import pickle
import json
import numpy as np

class Event:
    """Class to represent an event with its associated data.

    Attributes:
        run_number (int): an integer representing the run number.
        serial_number (int): an integer representing the serial number within the run
        n_mpmt (int): number of MPMTs in the event, n_mpmt = max(i_mpmt) + 1
        mpmt_status (boolean list): a list of booleans indicating the status of each MPMT.
        pmt_status (boolean list): a list of booleans indicating the status of each PMT, indexed by [i_mpmt][i_pmt].
        hit_times (float list): a list of floats representing the hit times for each PMT, indexed by [i_mpmt][i_pmt][i_hit].
        hit_charges (float list): a list of floats representing the hit charges for each PMT, indexed by [i_mpmt][i_pmt][i_hit].

    """
    def __init__(self, run_number, serial_number, n_mpmt):
        if not isinstance(run_number, int) or not isinstance(serial_number, int) or not isinstance(n_mpmt, int):
            raise TypeError("run_number, serial_number, and n_mpmt must be integers")
        self.run_number = run_number
        self.serial_number = serial_number
        if n_mpmt < 1:
            raise ValueError("n_mpmt must be at least 1")

        self.n_mpmt = n_mpmt
        self.npmt_per_mpmt = 19
        self.mpmt_status = [False for _ in range(n_mpmt)]
        self.pmt_status = [[False for _ in range(self.npmt_per_mpmt)] for _ in range(n_mpmt)]
        self.hit_times = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(n_mpmt)]
        self.hit_charges = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(n_mpmt)]

    def __repr__(self):
        return (f"Event(run_number={self.run_number}, serial_number={self.serial_number}, "
                f"n_mpmt={self.n_mpmt}, npmt_per_mpmt={self.npmt_per_mpmt})")

    def set_event_id(self, run_number, serial_number):
        """Set the run number and serial number of the event.

        Args:
            run_number (int): The run number to set.
            serial_number (int): The serial number to set.
        """
        if not isinstance(run_number, int) or not isinstance(serial_number, int):
            raise TypeError("run_number and serial_number must be integers")
        self.run_number = run_number
        self.serial_number = serial_number

    def reset_status(self):
        """Reset the status of all MPMTs and PMTs to False."""
        self.mpmt_status = [False for _ in range(self.n_mpmt)]
        self.pmt_status = [[False for _ in range(self.npmt_per_mpmt)] for _ in range(self.n_mpmt)]

    def set_mpmt_status(self, mpmts, status=True):
        """Set the status of specified MPMTs to status

        Args:
            mpmts (list): List of MPMT indices.
            status (bool): Boolean value to set the status of each MPMT in the list.
        """
        for i_mpmt in mpmts:
            self.mpmt_status[i_mpmt] = status

    def set_pmt_status(self, mpmt, pmts, status=True):
        """Set the status of specified PMTs in a given MPMT to status

        Args:
            mpmt (int): MPMT index.
            pmts (list): List of PMT indices.
            status (bool): Boolean value to set the status of each PMT in the list.
        """
        for i_pmt in pmts:
            self.pmt_status[mpmt][i_pmt] = status

    def clear_hits(self):
        """Clear all hit times and charges for the event."""
        self.hit_times = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(self.n_mpmt)]
        self.hit_charges = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(self.n_mpmt)]

    def add_hit_list(self,mpmts,pmts,times,charges):
        """Add hits to the event given lists of MPMT indices, PMT indices, times, and charges.

        Args:
            mpmts (list): List of MPMT indices.
            pmts (list): List of PMT indices.
            times (list): List of hit times.
            charges (list): List of hit charges.
        """
        for i in range(len(mpmts)):
            i_mpmt = mpmts[i]
            i_pmt = pmts[i]
            time = times[i]
            charge = charges[i]
            self.hit_times[i_mpmt][i_pmt].append(time)
            self.hit_charges[i_mpmt][i_pmt].append(charge)

    def get_combined_event(self, event_list, peak_time_align=2000, time_windows=((1500,2500),(-8,+4))):
        """Combine events from a list of Event objects.
        The mPMT and PMT status from the current event is used to determine which hits to include.
        The hit times from each event are histogrammed in 1ns bins within the specified time window to find the peak time.
        Each event's hit times are then shifted so that the peak time aligns with peak_time_align.
        Hits within the specified delta time window around the peak time are included in the combined event.

        Args:
            event_list (list): List of Event objects to average.
            peak_time_align: float: The peak time to align to. Time distribution with 1ns bins will be shifted so that 
            the bin with the most hits is at peak_time.
            time_windows: tuple: A tuple of two tuples specifying 
             0: The time window (min_time, max_time) to consider hits for finding peak time
             1: The time window (min_delta, max_delta) of hits for inclusion in the combined event
            
        Returns:
            Event: A new Event object with hits from all events included (withing the specified time windows).
            The event has the new attributes:
                avg_hit_times (float list): a list of floats representing the average hit times for each
                                            PMT, indexed by [i_mpmt][i_pmt]
                avg_hit_charges (float list): a list of floats representing the average hit charges for each
                                              PMT, indexed by [i_mpmt][i_pmt]
                rms_hit_times (float list): a list of floats representing the RMS of hit times for each
                                            PMT, indexed by [i_mpmt][i_pmt]
                rms_hit_charges (float list): a list of floats representing the RMS of hit charges for each
                                              PMT, indexed by [i_mpmt][i_pmt]
                frac_missing_hits (float list): a list of floats representing the fraction of events with no hit for each
                                                PMT, indexed by [i_mpmt][i_pmt]
                nus (float list): a list of floats representing the estimated expectation value of photoelectrons per
                                  event per PMT, indexed by [i_mpmt][i_pmt]. This is calculated as -ln(frac_missing_hits)
        """

        combined_event = self.copy() # Create a copy of the current event to hold combined hits
        combined_event.clear_hits() # Clear existing hits

        for ev in event_list:
            hit_times = []
            for i_mpmt in range(self.n_mpmt):
                if self.mpmt_status[i_mpmt]:
                    for i_pmt in range(self.npmt_per_mpmt):
                        if self.pmt_status[i_mpmt][i_pmt]:
                            hit_times.extend(ev.hit_times[i_mpmt][i_pmt])

            values, edges = np.histogram(hit_times, bins=np.arange(time_windows[0][0], time_windows[0][1], 1))
            peak_time = edges[np.argmax(values)]
            peak_shift = peak_time_align - peak_time
            min_time = peak_time_align + time_windows[1][0]
            max_time = peak_time_align + time_windows[1][1]

            for i_mpmt in range(self.n_mpmt):
                if self.mpmt_status[i_mpmt]:
                    for i_pmt in range(self.npmt_per_mpmt):
                        if self.pmt_status[i_mpmt][i_pmt]:
                            for i_hit in range(len(ev.hit_times[i_mpmt][i_pmt])):
                                hit_time = ev.hit_times[i_mpmt][i_pmt][i_hit]
                                hit_charge = ev.hit_charges[i_mpmt][i_pmt][i_hit]
                                shifted_time = hit_time + peak_shift
                                if min_time <= shifted_time <= max_time:
                                    combined_event.hit_times[i_mpmt][i_pmt].append(shifted_time)
                                    combined_event.hit_charges[i_mpmt][i_pmt].append(hit_charge)

        # Calculate average, RMS, fraction of missing hits, and nus for each PMT

        combined_event.avg_hit_times = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(self.n_mpmt)]
        combined_event.avg_hit_charges = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(self.n_mpmt)]
        combined_event.rms_hit_times = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(self.n_mpmt)]
        combined_event.rms_hit_charges = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(self.n_mpmt)]
        combined_event.frac_missing_hits = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(self.n_mpmt)]
        combined_event.nus = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(self.n_mpmt)]

        for i_mpmt in range(self.n_mpmt):
            if self.mpmt_status[i_mpmt]:
                for i_pmt in range(self.npmt_per_mpmt):
                    if self.pmt_status[i_mpmt][i_pmt]:
                        if len(combined_event.hit_times[i_mpmt][i_pmt]) > 0:
                            combined_event.avg_hit_times[i_mpmt][i_pmt].append(np.mean(combined_event.hit_times[i_mpmt][i_pmt]))
                            combined_event.avg_hit_charges[i_mpmt][i_pmt].append(np.mean(combined_event.hit_charges[i_mpmt][i_pmt]))
                            combined_event.rms_hit_times[i_mpmt][i_pmt].append(np.std(combined_event.hit_times[i_mpmt][i_pmt]))
                            combined_event.rms_hit_charges[i_mpmt][i_pmt].append(np.std(combined_event.hit_charges[i_mpmt][i_pmt]))
                        else:
                            combined_event.avg_hit_times[i_mpmt][i_pmt].append(0.0)
                            combined_event.avg_hit_charges[i_mpmt][i_pmt].append(0.0)
                            combined_event.rms_hit_times[i_mpmt][i_pmt].append(0.0)
                            combined_event.rms_hit_charges[i_mpmt][i_pmt].append(0.0)
                        n_events = len(event_list)
                        n_hits = len(combined_event.hit_times[i_mpmt][i_pmt])
                        frac_missing_hits = (n_events - n_hits) / n_events
                        combined_event.frac_missing_hits[i_mpmt][i_pmt].append(frac_missing_hits)
                        if frac_missing_hits >= 1.0:
                            combined_event.nus[i_mpmt][i_pmt].append(0.0)
                        elif frac_missing_hits <= 0.0:
                            combined_event.nus[i_mpmt][i_pmt].append(float('inf'))
                        else:
                            combined_event.nus[i_mpmt][i_pmt].append(-np.log(frac_missing_hits))

        return combined_event


    def copy(self):
        """Create a deep copy of the Event object.

        Returns:
            Event: A deep copy of the current Event object.
        """
        return pickle.loads(pickle.dumps(self))

    def save(self, filename):
        """Save the Event object to a file using pickle.

        Args:
            filename (str): The name of the file to save the object to.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load an Event object from a file using pickle.

        Args:
            filename (str): The name of the file to load the object from.

        Returns:
            Event: The loaded Event object.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def to_json(self):
        """Convert the Event object to a JSON string.

        Returns:
            str: The JSON string representation of the Event object.
        """
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def from_json(json_str):
        """Create an Event object from a JSON string.

        Args:
            json_str (str): The JSON string representation of the Event object.

        Returns:
            Event: The created Event object.
        """
        data = json.loads(json_str)
        ev = Event(data['run_number'], data['serial_number'], data['n_mpmt'])
        ev.npmt_per_mpmt = data['npmt_per_mpmt']
        ev.mpmt_status = data['mpmt_status']
        ev.pmt_status = data['pmt_status']
        ev.hit_times = data['hit_times']
        ev.hit_charges = data['hit_charges']
        return ev

class SimulatedEvent(Event):
    """Class to represent a simulated event, inheriting from Event.

    Attributes:
        emitters (list): A list of emitters that produced the simulated event
        expected_hit_times (float list): a list of floats representing the expected hit times for each PMT
                                         having >0 expected PE from emitters, indexed by [i_mpmt][i_pmt][i_hit].
        expected_hit_pe (float list): a list of floats representing the expected number of PE for each PMT,
                                      indexed by [i_mpmt][i_pmt][i_hit].
        true_hit_pe (float list): a list of floats representing the true number of PE for each PMT,
                                      indexed by [i_mpmt][i_pmt][i_hit].
        noise_hit_times (float list): a list of floats representing the noise hit times for each PMT,
                                      indexed by [i_mpmt][i_pmt][i_noise_hit].
        noise_hit_pe (float list): a list of floats representing the noise hit pe (typically 1)
    """
    def __init__(self, run_number, serial_number, n_mpmt):
        super().__init__(run_number, serial_number, n_mpmt)
        self.emitters = []
        self.expected_hit_times = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(n_mpmt)]
        self.expected_hit_pe = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(n_mpmt)]
        self.true_hit_pe = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(n_mpmt)]
        self.noise_hit_times = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(n_mpmt)]
        self.noise_hit_pe = [[[] for _ in range(self.npmt_per_mpmt)] for _ in range(n_mpmt)]

    def __repr__(self):
        return (f"SimulatedEvent(run_number={self.run_number}, serial_number={self.serial_number}, "
                f"n_mpmt={self.n_mpmt}, npmt_per_mpmt={self.npmt_per_mpmt}, "
                f"n_emitters={len(self.emitters)})")

    @staticmethod
    def from_json(json_str):
        """Create a SimulatedEvent object from a JSON string.

        Args:
            json_str (str): The JSON string representation of the SimulatedEvent object.

        Returns:
            SimulatedEvent: The created SimulatedEvent object.
        """
        data = json.loads(json_str)
        ev = SimulatedEvent(data['run_number'], data['serial_number'], data['n_mpmt'])
        ev.npmt_per_mpmt = data['npmt_per_mpmt']
        ev.mpmt_status = data['mpmt_status']
        ev.pmt_status = data['pmt_status']
        ev.hit_times = data['hit_times']
        ev.hit_charges = data['hit_charges']
        ev.emitters = data['emitters']
        ev.expected_hit_times = data['expected_hit_times']
        ev.expected_hit_pe = data['expected_hit_pe']
        ev.noise_hit_times = data['noise_hit_times']
        ev.noise_hit_pe = data['noise_hit_pe']
        return ev