import scipy.io as sio
import numpy as np
import pandas as pd
import data_helpers

# outputs - Rear & front cornering, long, combined data cleaned and classified

def main():
    # Normal load sweeps
    # Got from 1965 Summary Tables
    # Normal Loads for Cornering
    L1 = np.array([-50, -100, -150, -200, -250]) / 0.224809
    L2 = np.array([-50, -100, -150, -200, -250, -350]) / 0.224809
    L3 = np.array([-50, -100, -150, -250, -350]) / 0.224809 # This is our L3
    L4 = np.array([-350, -150, -250, -50]) / 0.224809 # This is our L4
    # L3 = np.array([-50, -150, -250]) / 0.224809 # Orignally their L3
    # L4 = np.array([-50, -100, -150, -200, -250]) / 0.224809 # Orignally their L4
    # L6 = np.array([-50, -150, -200, -250]) / 0.224809 # We just dont have this much 
    # L7 = np.array([-50, -150, -250, -350]) / 0.224809 # We just dont have this much 
    L5= np.array([-50, -100, -150, -200, -250, -50, -100, -150, -200, -250, -350]) / 0.224809
    
    # Camber sweeps
    l1 = np.array([0, 2, 4])
    l2 = np.array([0, -4, 4, 0])
    l5 = np.array([0, 2, 4, 0, -4, 4, 0])
    
    # velocity sweeps
    V1 = np.array([0, 25, 2]) * 1.60934
    V3 = np.array([25, 15, 45]) * 1.60934
    V25 = np.array([25]) * 1.60934
    V5 = np.array([25, 25, 15, 45]) * 1.60934
    
    # pressure sweep - this is the exact same for UIC!!!
    P = np.array([8, 10, 12, 14]) * 6.89476 # includes P1r and P2r
    
    # slip angle sweep (for S1)
    S1 = np.array([-1, 1, 6])
    S2 = np.array([-4, 12, -12, 4])
    # slip angle sweep (for long/combined data)
    S4 = np.array([0, -1, -6])
    
    def create_sweep_dict(normal_load, camber, pressure, velocity, slip_angle = None):
        if slip_angle is not None:
            return {"load" : {"sweep" : normal_load, "label" : "FZ" },
                    "camber" : {"sweep" : camber, "label" : "IA"},
                    "pressure" : {"sweep" : pressure, "label" : "P"},
                    "velocity" : {"sweep" : velocity, "label" : "V"},
                    "slip" : {"sweep" : slip_angle, "label" : "SA"}}
        else:
            return {"load" : {"sweep" : normal_load, "label" : "FZ" },
                    "camber" : {"sweep" : camber, "label" : "IA"},
                    "pressure" : {"sweep" : pressure, "label" : "P"},
                    "velocity" : {"sweep" : velocity, "label" : "V"}}

    output_directory = "tire_data/processed_data/"

    data_map = {#"cornering_hoosier_r25b_18x7-5_10x7_run1": {"file_path" : "tire_data/run_data/Round6/B1654run21.mat", 
                #                                            "sweeps" : create_sweep_dict(L4, l2, P, V1), "avg": True},
                
                #"cornering_hoosier_r25b_18x7-5_10x7_run2": {"file_path" : "tire_data/run_data/Round6/B1654run22.mat", 
                #                                            "sweeps" : create_sweep_dict(L4, l2, P, V3), "avg": True}, 
                
                # "cornering_hoosier_r25b_18x6_10x7": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_Cornering_Matlab_SI_Round6/B1654run27.mat"],
                # "sweeps" : cornering_variable_sweeps, "avg": True},

                # "cornering_hoosier_r25b_18x6_10x6": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_Cornering_Matlab_SI_Round6/B1654run29.mat"],
                # "sweeps" : cornering_variable_sweeps, "avg": True},

                # "braking_hoosier_r25b_18x7-5_10x8": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_DriveBrake_Matlab_SI_Round6/B1654run38.mat",
                # "tire_data/raw_data/RunData_10inch_DriveBrake_Matlab_SI_Round6/B1654run39.mat"],
                # "sweeps" : braking_variable_sweeps, "avg" : False},
                
                # "braking_hoosier_r25b_18x6_10x7": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_DriveBrake_Matlab_SI_Round6/B1654run43.mat"],
                # "sweeps" : braking_variable_sweeps, "avg" : False},

                # "braking_hoosier_r25b_18x6_10x6": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_DriveBrake_Matlab_SI_Round6/B1654run41.mat"],
                # "sweeps" : braking_variable_sweeps, "avg" : False},

                # "cornering_hoosier_r25b_18x7-5_10x8": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_Cornering_Matlab_SI_Round6/B1654run24.mat",
                # "tire_data/raw_data/RunData_10inch_Cornering_Matlab_SI_Round6/B1654run25.mat"], "sweeps" : cornering_variable_sweeps, "avg": True},
                
                # "cornering_hoosier_LCO_18x6_10x7": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_Cornering_Matlab_SI_Round6/B1654run33.mat"],
                # "sweeps" : cornering_variable_sweeps, "avg": True},

                # "cornering_hoosier_LCO_18x6_10x6": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_Cornering_Matlab_SI_Round6/B1654run31.mat"],
                # "sweeps" : cornering_variable_sweeps, "avg": True},
                
                #"braking_hoosier_r25b_18x7-5_10x7_run1": {"file_path" : "tire_data/run_data/Round6/B1654run35.mat",
                #"sweeps" : create_sweep_dict(L6, l2, P, V1, S4), "avg" : False},
                
                #"braking_hoosier_r25b_18x7-5_10x7_run2": {"file_path" : "tire_data/run_data/Round6/B1654run36.mat",
                #"sweeps" : create_sweep_dict(L6, l2, P, V3, S4), "avg" : False},
                
                # "braking_hoosier_LCO_18x6_10x7": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_DriveBrake_Matlab_SI_Round6/B1654run47.mat"],
                # "sweeps" : braking_variable_sweeps, "avg" : False},

                # "braking_hoosier_LCO_18x6_10x6": {"data_file_names" : ["tire_data/raw_data/RunData_10inch_DriveBrake_Matlab_SI_Round6/B1654run45.mat"],
                # "sweeps" : braking_variable_sweeps, "avg" : False},
                
                # "cornering_hoosier_r25b_16x6_10x6": {"data_file_names" : ["tire_data/raw_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run9.mat",
                # "tire_data/raw_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run10.mat"],
                # "sweeps" : cornering_variable_sweeps, "avg": True},
                
                # "cornering_hoosier_r25b_16x6_10x7" : {"data_file_names" : ["tire_data/raw_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run12.mat",
                # "tire_data/raw_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run13.mat"],
                # "sweeps" : cornering_variable_sweeps, "avg": True},
                
                # "cornering_hoosier_r25b_16x7-5_10x8" : {"data_file_names" : ["tire_data/raw_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run6.mat"],
                # "sweeps" : cornering_variable_sweeps, "avg": True},

                # "camber_hoosier_r25b_16x7-5_10x8" : {"data_file_names" : ["tire_data/raw_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run7.mat"],
                # "sweeps" : cornering_camber, "avg": True},
                
                #"cornering_hoosier_r25b_16x7-5_10x7_run1" : {"file_path" : "tire_data/run_data/Round8/B1965run2.mat",
                #"sweeps" : create_sweep_dict(L1, l2, P, V1), "avg": True},

                #"cornering_hoosier_r25b_16x7-5_10x7_run2" : {"file_path" : "tire_data/run_data/Round8/B1965run4.mat",
                #"sweeps" : create_sweep_dict(L2, l2, P, V3), "avg": True},

                #"cornering_hoosier_r25b_20-5x7_13x7_run1": {"file_path" : "tire_data/run_data/Round6/B1654run9.mat",
                #"sweeps" : create_sweep_dict(L2, l2, P, V1), "avg": True},
                
                #"braking_hoosier_r25b_20-5x7_13x7_run1": {"file_path" : "tire_data/run_data/Round6/B1654run55.mat",
                #"sweeps" : create_sweep_dict(L7, l2, P, V1, S4), "avg" : False}
        
                #Attempt at adding UIC data to this project using what was uncommented before
                # Future Ass reference - Formula Car Tires: Hosier 16x7 LC0 8" rim width - 43075 16x7.5-10 LCO - 8
                "cornering_hoosier_r25b_16x7-5_10x7_run1" : {"file_path" : ["tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run2.mat", "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run4.mat"],
                "sweeps" : create_sweep_dict(L1, l1, P, V25), "avg": True},

                "cornering_hoosier_r25b_16x7-5_10x7_run2" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run4.mat",
                "sweeps" : create_sweep_dict(L2, l2, P, V3), "avg": True},

                "cornering_hoosier_r25b_16x7-5_10x8_run1" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run6.mat",
                "sweeps" : create_sweep_dict(L1, l1, P, V25), "avg": True},

                "cornering_hoosier_r25b_16x7-5_10x8_run2" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run7.mat",
                "sweeps" : create_sweep_dict(L2, l2, P, V3), "avg": True},

                "cornering_hoosier_r25b_16x6-0_10x6_run1" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run9.mat",
                "sweeps" : create_sweep_dict(L1, l1, P, V25), "avg": True},

                "cornering_hoosier_r25b_16x6-0_10x6_run2" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run10.mat",
                "sweeps" : create_sweep_dict(L2, l2, P, V3), "avg": True},

                "cornering_hoosier_r25b_16x6-0_10x7_run1" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run12.mat",
                "sweeps" : create_sweep_dict(L1, l1, P, V25), "avg": True},

                "cornering_hoosier_r25b_16x6-0_10x7_run2" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run13.mat",
                "sweeps" : create_sweep_dict(L2, l2, P, V3), "avg": True},

                "cornering_hoosier_lc10_16x7-5_10x8_run1" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run15.mat",
                "sweeps" : create_sweep_dict(L1, l1, P, V25), "avg": True},

                "cornering_hoosier_lc10_16x7-5_10x8_run2" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run16.mat",
                "sweeps" : create_sweep_dict(L2, l2, P, V3), "avg": True},

                "cornering_hoosier_lc10_16x7-5_10x7_run1" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run18.mat",
                "sweeps" : create_sweep_dict(L1, l1, P, V25), "avg": True},

                "cornering_hoosier_lc10_16x7-5_10x7_run2" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run19.mat",
                "sweeps" : create_sweep_dict(L2, l2, P, V3), "avg": True},

                "cornering_hoosier_lc10_16x6-0_10x6_run1" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run21.mat",
                "sweeps" : create_sweep_dict(L1, l1, P, V25), "avg": True},

                "cornering_hoosier_lc10_16x6-0_10x6_run2" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run22.mat",
                "sweeps" : create_sweep_dict(L2, l2, P, V3), "avg": True},

                "cornering_hoosier_lc10_16x6-0_10x7_run1" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run24.mat",
                "sweeps" : create_sweep_dict(L1, l1, P, V25), "avg": True},

                "cornering_hoosier_lc10_16x6-0_10x7_run2" : {"file_path" : "tire_data/run_data/RunData_Cornering_Matlab_SI_10inch_Round8/B1965run25.mat",
                "sweeps" : create_sweep_dict(L2, l2, P, V3), "avg": True}
            }

    for output_name, data_info in data_map.items():
        # load matlab file and convert to pandas df
        ## NOTEE - if multiple sweeps call the same matlab file, this can cause this to stop working, dont do that
        df = data_helpers.import_data(sio.loadmat(data_info["file_path"]), run_data = True)

        # classify sweeps on data
        for variable, info in data_info["sweeps"].items():
            temp_nearest_func = lambda x: get_nearest_value(info["sweep"], x)
            df[variable] = df[info["label"]].apply(temp_nearest_func)

        # period of oscillation is ~ 10.5 data points, remove oscillation
        # TODO: Use FFT to find oscillation period, and remove it
        if data_info["avg"]:
            for target_var in ["FY", "FX", "FZ"]:
                df[target_var] = moving_average(df[target_var], 10)

        # export data back to matlab file
        data_helpers.export_dataframe_to_mat(f'{output_directory}{output_name}.mat', df)



def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

if __name__ == "__main__":
    main()