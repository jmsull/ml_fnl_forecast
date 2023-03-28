# Most of the information/features calculated here are not used as the input features for the machine learning in the paper

import numpy as np
import pandas as pd
import h5py
import scipy.spatial as spatial
from sklearn.decomposition import PCA   # end up not using
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RectBivariateSpline
import groupcat
from input_features import Calculate_features, Get_galaxy_position, Get_halo_properties, New_feature_names
from interpolation_data import *
import os

# SELECT REDSHIF
catalog_number = 50  # 50 for z = 1s

PATH_TO_SAVE = "data/datasets/"


def Load_TNG():
    """Load halo and galaxy (subhalo) data from IllustrisTNG-300 catalog, and halo concentration from supplementary catalog."""
    path_catalog = "../data/TNG-300"
    path_supplementary = f"../data/TNG-300/supplementary_catalog/halo_structure_0{catalog_number}.hdf5"
    data = groupcat.load(path_catalog, catalog_number)
    subhalos = data["subhalos"]
    halos = data["halos"]
    # Load halo concentration from supplementary catalog
    with h5py.File(path_supplementary, 'r') as f:
        halo_concentration = list(f["c200c"])
    return subhalos, halos, halo_concentration


def New_z(z, vel_z):
    """Transformation from real to redshift space at z=1"""
    H = 0.178
    a = 0.5  # 1/(1+z) ; z = 1
    new = z + vel_z/(a*H)
    return new


def Illustris_features(subhalos, halos, halo_concentration, redshift_space=True):
    """Get all features that we are potentially interested in from illustrisTNG catalog and convert them to pandas data frame."""
    # subhalos, halos, halo_concentration = load_TNG()
    galaxy_flag = subhalos["SubhaloFlag"]
    lum_z = subhalos["SubhaloStellarPhotometrics"][:, -1]
    lum_r = subhalos["SubhaloStellarPhotometrics"][:, -3]
    lum_g = subhalos["SubhaloStellarPhotometrics"][:, -4]
    mass = subhalos["SubhaloMassType"][:, -2]
    sfr = subhalos["SubhaloSFR"]
    max_vel = subhalos["SubhaloVmax"]
    rad = subhalos["SubhaloHalfmassRadType"][:, -2]
    v_disp = subhalos["SubhaloVelDisp"]
    halo_index = subhalos["SubhaloGrNr"]
    z_vel = subhalos["SubhaloVel"][:, -2]
    crit200 = halos["Group_M_Crit200"][halo_index]
    crit500 = halos["Group_M_Crit500"][halo_index]
    mean200 = halos["Group_M_Mean200"][halo_index]
    GroupVel = halos["GroupVel"][halo_index]
    GroupVel = np.sqrt(GroupVel[:, 0]**2 +
                       GroupVel[:, 1]**2 + GroupVel[:, 2]**2)
    # Addional information
    coordinates = subhalos["SubhaloPos"]
    posX = coordinates[:, 0]
    posY = coordinates[:, 1]
    posZ = coordinates[:, 2]
    # trasnformation to redshift space
    if redshift_space:
        new_pos_z = [New_z(posZ[i], z_vel[i]) for i in range(len(posZ))]
    coordinates = np.vstack([posX, posY, new_pos_z]).T

    # number of instnances, needed to (later) get central halo
    ID = range(len(subhalos["SubhaloSFR"]))

    halo_central_galaxy = halos["GroupFirstSub"][halo_index]
    halo_concentration = np.array(halo_concentration)[halo_index]
    halo_mass = np.log10(np.multiply(
        halos["Group_M_Mean200"], 10**10))[halo_index]

    # additional halo properties

    halo_center = halos["GroupPos"][halo_index]
    halo_x = halo_center[:, 0]
    halo_y = halo_center[:, 1]
    halo_z = halo_center[:, 2]

    features = ["ID", "lum_z", "lum_r", "lum_g", "mass", "sfr", "max_vel", "rad", "v_disp2", "posX", "posY", "posZ",
                "halo_x", "halo_y", "halo_z", "central_gal", "halo_mass", "halo_conc", "halo_index", "crit200", "crit500", "mean200", "z_vel", "group_vel"]
    # MAKE DATASET
    dataset = pd.DataFrame(np.array([ID, lum_z, lum_r, lum_g, mass, sfr, max_vel, rad, v_disp, posX, posY, new_pos_z, halo_x,
                                     halo_y, halo_z, halo_central_galaxy, halo_mass, halo_concentration, halo_index,
                                     crit200, crit500, mean200, z_vel, GroupVel]).transpose(), columns=features)

    not_galaxies = np.array(np.where(galaxy_flag == False)[
                            0])  # Indexes of non galaxies
    coordinates = np.delete(coordinates, not_galaxies, axis=0)
    dataset.drop(not_galaxies, axis=0, inplace=True)
    dataset["v_disp2"] = np.square(dataset["v_disp2"]).reset_index(drop=True)
    return dataset, coordinates


def Sample_selection(sample_size, sample_type, subhalos, halos, halo_concentration, sSFR, return_interpol=False, max_conc=1000):
    """Select ELG, LRG  sample of galaxies, or galaxy sample based on their luminosity."""
    # Parse relevant features from subhalos, halos, and halo_concentration
    data, coordinates = Illustris_features(subhalos, halos, halo_concentration)
    # Drop data that doesn't satisfy conditions: non negative mass, non negative sfr, non negative halo mass,
    # known (and finite) halo concentration (those are non-physical and may appear as a numerical errors)
    valid_data = (data["mass"] > 0) & (data["sfr"] > 0) & (data["halo_mass"] > 0) & (
        data["halo_conc"] < max_conc) & (np.isfinite(data["halo_conc"])) & (np.isfinite(data["halo_mass"]))
    coordinates = coordinates[valid_data]
    data = data[valid_data].reset_index(drop=True)
    data["halo_conc_log"] = np.log(data["halo_conc"])

    # Compute specific star formation rate (ssfr) and drop non-numerical values
    # multiply to convert units
    data["ssfr"] = np.log10(
        np.divide(data["sfr"], np.multiply(data["mass"], 10**10)))  # unit conversion
    valid_data = np.where(np.isfinite(data["ssfr"]))[0]
    coordinates = coordinates[valid_data]
    data = data.iloc[valid_data].reset_index(drop=True)
    if return_interpol == True:
        bright_galaxies = np.argpartition(
            data["lum_z"], sample_size)[:sample_size]
        return data.iloc[bright_galaxies].reset_index(drop=True), coordinates[bright_galaxies]
    # Otherwise, select galaxies based on their type and brightness
    assert sample_type in ["ELG", "LRG", "lum_z", "lum_r", "lum_g"]

    if sample_type in ["lum_z", "lum_r", "lum_g"]:
        bright_galaxies = np.argpartition(
            data[sample_type], sample_size).to_numpy()[:sample_size]
    else:
        if sample_type == "ELG":
            coordinates = coordinates[data["ssfr"] > -sSFR]  # 9.23
            data = data[data["ssfr"] > -sSFR]
        else:  # sample_type == "LRG"
            coordinates = coordinates[data["ssfr"] < -sSFR]  # 9.23
            data = data[data["ssfr"] < -sSFR]
        # Select the heaviest galaxies based on the specified number of galaxies
        bright_galaxies = np.argpartition(
            data["mass"], -sample_size).to_numpy()[-sample_size:]
    # Return the selected dataset and coordinates
    coordinates = coordinates[bright_galaxies]
    data = data.iloc[bright_galaxies].reset_index(drop=True)
    return data, coordinates


def Processed_data_with_features(sample_size, sample_type, subhalos, halos, halo_concentration, N_additional, type_additional, sSFR, max_conc=100):
    """Get the samples along with additional sample that may be used only to gather additional information about (local) halo environment. Calculate all the features."""
    # do not include additional environmental information, just use base dataset.
    if N_additional == 0:
        dataset, coordinates = Sample_selection(
            sample_size, sample_type, subhalos, halos, halo_concentration, sSFR, max_conc=max_conc)
        indexes = dataset.index
    else:   # dataset with additional information from differently selected dataset
        dataset, coordinates = Sample_selection(
            sample_size, sample_type, subhalos, halos, halo_concentration, sSFR, max_conc=max_conc)
        dataset_add, coordinates_add = Sample_selection(
            N_additional, type_additional, subhalos, halos, halo_concentration, sSFR, max_conc=max_conc)
        indexes = dataset.index

        org_ids = list(dataset["ID"].values)
        to_delete = []
        for i in dataset_add.index:
            if dataset_add["ID"].iloc[i] in org_ids:
                to_delete.append(i)
        coordinates_add = np.delete(coordinates_add, i, axis=0)
        dataset_add.drop(i, axis=0, inplace=True)
        # Add additional and origial
        coordinates = np.vstack([coordinates, coordinates_add])
        dataset_add.index = range(
            dataset.shape[0], dataset.shape[0]+dataset_add.shape[0])
        dataset = pd.concat([dataset, dataset_add], axis=0,
                            ignore_index=True).reset_index(drop=True)
    kd_tree = spatial.KDTree(coordinates)

    new_dataset = []
    for galaxy in indexes:
        ID = [int(dataset["ID"].iloc[galaxy])]
        galaxy_position = Get_galaxy_position(dataset, galaxy)
        halo_properties = Get_halo_properties(dataset, galaxy)
        features = Calculate_features(galaxy, coordinates, kd_tree, dataset)
        new_dataset.append(list(ID+features+galaxy_position+halo_properties))
        feature_names, features_to_standardize = New_feature_names()
    new_dataset = pd.DataFrame(new_dataset, columns=feature_names)

    new_dataset["raw_mass"] = new_dataset["mass"].copy()
    new_dataset["sum_m"] = np.log10(new_dataset["sum_m"])
    new_dataset["mass"] = np.log10(new_dataset["mass"])
    # needed for some plots

    # scaler = StandardScaler()   # standardize features
    # new_dataset[features_to_standardize] = scaler.fit_transform(
    #     new_dataset[features_to_standardize])

    # We have tested that transofrming any data with PCA worsen results so we do not include it here.
    return new_dataset


def Sort_into_mass_bins(data, bins_mean_m, feature_name="halo_mass"):
    """Sort the dataframe into 4 mass bins, need to interpolate bphi and b1 from Lazeyras."""
    half_width = bins_mean_m[1] - bins_mean_m[0]
    right_edges = [i+half_width for i in mass_bins]
    ind1, ind2, ind3, ind4 = [], [], [], []
    data.index = range(data.shape[0])
    for i in data.index:
        if data[feature_name].iloc[i] < right_edges[0]:
            ind1.append(i)
        elif data[feature_name].iloc[i] < right_edges[1]:
            ind2.append(i)
        elif data[feature_name].iloc[i] < right_edges[2]:
            ind3.append(i)
        elif data[feature_name].iloc[i] < right_edges[3]:
            ind4.append(i)
    data1 = data.iloc[ind1].copy()
    data2 = data.iloc[ind2].copy()
    data3 = data.iloc[ind3].copy()
    data4 = data.iloc[ind4].copy()
    return data1, data2, data3, data4


def Split_into_tertiles(data, feature="halo_conc_log"):
    """Split data into tertiles based on given feature. Need for interpolation of bphi and b1."""
    sorted_index = np.argsort(data[feature].values)  # based on mass
    data1 = data.iloc[sorted_index[:len(sorted_index)//3]]
    data2 = data.iloc[sorted_index[len(
        sorted_index)//3:2*len(sorted_index)//3]]
    data3 = data.iloc[sorted_index[-len(sorted_index)//3:]]
    return data1, data2, data3


def Interpolate_Fig1(subhalos, halos, halo_concentration, sSFR, max_conc=100):
    """Interpolation from Lazeyras. A bit messy code but is tested and is working correctly."""
    dataset, coordinates = Sample_selection(
        INTERPOLATED_N, "lum_z", subhalos, halos, halo_concentration, sSFR, return_interpol=True, max_conc=max_conc)
    data1, data2, data3, data4 = Sort_into_mass_bins(
        dataset, mass_bins, feature_name="halo_mass")

    data1_c1, data1_c2, data1_c3 = Split_into_tertiles(
        data1, feature="halo_conc_log")
    data2_c1, data2_c2, data2_c3 = Split_into_tertiles(
        data2, feature="halo_conc_log")
    data3_c1, data3_c2, data3_c3 = Split_into_tertiles(
        data3, feature="halo_conc_log")
    data4_c1, data4_c2, data4_c3 = Split_into_tertiles(
        data4, feature="halo_conc_log")

    conc_low = [np.mean(data1_c1["halo_conc_log"]), np.mean(data2_c1["halo_conc_log"]), np.mean(
        data3_c1["halo_conc_log"]), np.mean(data4_c1["halo_conc_log"])]
    conc_medium = [np.mean(data1_c2["halo_conc_log"]), np.mean(data2_c2["halo_conc_log"]), np.mean(
        data3_c2["halo_conc_log"]), np.mean(data4_c2["halo_conc_log"])]
    conc_high = [np.mean(data1_c3["halo_conc_log"]), np.mean(data2_c3["halo_conc_log"]), np.mean(
        data3_c3["halo_conc_log"]), np.mean(data4_c3["halo_conc_log"])]

    M_y = np.array([11.79, 12.54, 13.28, 14.04])
    c_x = np.linspace(-0.3, 3.3)
    bbox = [-0.5, 3.5, 11.2, 14.7]

    # interpolate b_phi
    z_xy = []
    for c_i in c_x:
        ci_at_m = []
        for i in range(4):
            if c_i > conc_medium[i]:
                b_i = (b_medium[i]*(conc_high[i] - c_i) + b_high[i] *
                       (c_i - conc_medium[i]))/(conc_high[i] - conc_medium[i])
            else:
                b_i = (b_low[i]*(conc_medium[i] - c_i) + b_medium[i]
                       * (c_i - conc_low[i]))/(conc_medium[i] - conc_low[i])
            ci_at_m.append(b_i)
        z_xy.append(ci_at_m)

    int_func_b_phi = RectBivariateSpline(
        c_x, M_y, z_xy, bbox=bbox)    # interpolated function

    # interpolate b_1
    z_xy = []
    for c_i in c_x:
        ci_at_m = []
        for i in range(4):
            if c_i > conc_medium[i]:
                b_i = (b_medium_1[i]*(conc_high[i] - c_i) + b_high_1[i] *
                       (c_i - conc_medium[i]))/(conc_high[i] - conc_medium[i])
            else:
                b_i = (b_low_1[i]*(conc_medium[i] - c_i) + b_medium_1[i]
                       * (c_i - conc_low[i]))/(conc_medium[i] - conc_low[i])
            ci_at_m.append(b_i)
        z_xy.append(ci_at_m)

    int_func_b1 = RectBivariateSpline(
        c_x, M_y, z_xy, bbox=bbox)    # interpolated function

    return int_func_b_phi, int_func_b1


def Assign_b(data, int_func, new_features):
    """Assign either b_1 or b_phi to each halo from the dataset (based on its concentration and mass)"""
    b_phi_fitted = []
    data.index = range(data.shape[0])
    for i in data.index:
        b_phi_fitted.append(
            float(int_func(data["halo_conc_log"].iloc[i], data["halo_mass"].iloc[i])))
    data[new_features] = b_phi_fitted
    return data


def main(sample_size, sample_type, subhalos, halos, halo_concentration, N_additional=0, type_additional="ELG", sSFR=9.086, max_conc=100):
    """Extract all relevant data from TNG catalogue and make desired sample, which can be either mimic of ELG/LRG-like galaxies 
    (in which case also specify sSFR limit), or sample based on selected luminosity band"""
    dataset = Processed_data_with_features(sample_size, sample_type, subhalos, halos, halo_concentration,
                                           N_additional, type_additional, sSFR, max_conc=max_conc)
    dataset = dataset[dataset["halo_conc"] < max_conc]
    int_func_b_phi, int_func_b1 = Interpolate_Fig1(
        subhalos, halos, halo_concentration, sSFR, max_conc)

    dataset = Assign_b(dataset, int_func_b_phi, "b_phi")
    dataset = Assign_b(dataset, int_func_b1, "b_1")
    return dataset


def train_test(data, val=True):
    """Drop the galaxies near the edges of the simulation. 
    Split data into train test, validation subsets based on 'x' spatial position. The split is 70-20-10%. 
    If val==False we return train-test split in the 80-20 ratio."""

    # Drop data near edges of simulation (units of kpc/h)
    data = data[(data["posX"] > 1_000) & (data["posX"] < 204_000) &
                (data["posY"] > 1_000) & (data["posY"] < 204_000) & (data["posZ"] > 1_000) & (data["posZ"] < 204_000)]

    # x 'legnth' of simulation now is 203k kpc/h -> make train, test, validation splits:
    test = data[data["posX"] < 40_600].copy()  # 20%
    if val == True:
        validation = data[(data["posX"] > 40_600) & (
            data["posX"] < 60_900)].copy()  # 10%
        train = data[data["posX"] > 60_900].copy()  # 70%
        return train, test, validation
    else:
        train = data[data["posX"] > 40_600].copy()  # 70%
        return train, test


def save_all():
    import warnings
    # Warning because some galaxies (those are not of orur interest have mass/ halo concentration 0, which we logarithmize)
    warnings.filterwarnings("ignore")
    # Load data
    subhalos, halos, halo_concentration = Load_TNG()
    ELG5_908 = main(5*861, "ELG", subhalos, halos, halo_concentration,
                    sSFR=9.086, max_conc=100, type_additional="LRG", N_additional=2*861)
    ELG7_908 = main(7*861, "ELG", subhalos, halos, halo_concentration,
                    sSFR=9.086, max_conc=100, type_additional="LRG", N_additional=2*861)
    ELG10_908 = main(10*861, "ELG", subhalos, halos, halo_concentration,
                     sSFR=9.086, max_conc=100, type_additional="LRG", N_additional=2*861)

    ELG5_923 = main(5*861, "ELG", subhalos, halos, halo_concentration,
                    sSFR=9.23, max_conc=100, type_additional="LRG", N_additional=2*861)
    ELG7_923 = main(7*861, "ELG", subhalos, halos, halo_concentration,
                    sSFR=9.23, max_conc=100, type_additional="LRG", N_additional=2*861)
    ELG10_923 = main(10*861, "ELG", subhalos, halos, halo_concentration,
                     sSFR=9.23, max_conc=100, type_additional="LRG", N_additional=2*861)

    LRG_908 = main(2*861, "ELG", subhalos, halos, halo_concentration,
                   sSFR=9.086, max_conc=100, type_additional="ELG", N_additional=5*861)
    LRG_923 = main(2*861, "ELG", subhalos, halos, halo_concentration,
                   sSFR=9.23, max_conc=100, type_additional="ELG", N_additional=5*861)

    # save as csv
    ELG5_908.to_csv(os.path.join(
        PATH_TO_SAVE, "ELG/n5e-4_ssfr908.csv"), index=False)
    ELG7_908.to_csv(os.path.join(
        PATH_TO_SAVE, "ELG/n7e-4_ssfr908.csv"), index=False)
    ELG10_908.to_csv(os.path.join(
        PATH_TO_SAVE, "ELG/n10e-4_ssfr908.csv"), index=False)
    ELG5_923.to_csv(os.path.join(
        PATH_TO_SAVE, "ELG/n5e-4_ssfr923.csv"), index=False)
    ELG7_923.to_csv(os.path.join(
        PATH_TO_SAVE, "ELG/n7e-4_ssfr923.csv"), index=False)
    ELG10_923.to_csv(os.path.join(
        PATH_TO_SAVE, "ELG/n10e-4_ssfr923.csv"), index=False)

    LRG_908.to_csv(os.path.join(
        PATH_TO_SAVE, "LRG/n2e-4_ssfr908.csv"), index=False)
    LRG_923.to_csv(os.path.join(
        PATH_TO_SAVE, "LRG/n2e-4_ssfr923.csv"), index=False)


if __name__ == "__main__":
    save_all()
