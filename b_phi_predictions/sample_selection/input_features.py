import numpy as np

# Here we calculate a lot of properties regarding the galaxy and its environment, but do not use all in the paper.


def Neighbors_within_R(position, kd_tree, Rs=[0.25, 0.5, 1, 2, 3, 4, 5]):
    """Calculate number of neighbor galaxies within given galaxy."""
    neighbors = list()
    for R in Rs:
        R *= 1000  # to change units, from Mpc/h -> kpc/h
        number = len(kd_tree.query_ball_point(position, R, p=2))
        neighbors.append(number)  # -1 to exclude self galaxy
    return neighbors


def Anisotropy(position, kd_tree, coordinates, R):  # position
    """Metric of anisotropy of local environment around the given galaxy."""
    neighbors_within = kd_tree.query_ball_point(position, R, p=2.0)
    vectors = [coordinates[neighbor] -
               position for neighbor in neighbors_within]
    distances = [float(np.linalg.norm(v)) for v in vectors]
    vect = [0., 0., 0.]
    for i in range(len(vectors)):
        if distances[i] != 0:  # otherwise skip, as it does not contribute to anisotropy
            vect = np.add(vect, np.divide(vectors[i], distances[i]))
    return [np.linalg.norm(vect)]


def Galaxy_properties(galaxy, dataset):
    """Get 3 luminosity bands (which DESI survey is able to obseve), stellar mass, and half mass stellar radius of the galaxy."""
    properties = ["lum_z", "lum_r", "lum_g", "mass", "rad"]
    return [dataset[i].iloc[galaxy] for i in properties]

# def Galaxy_properties(galaxy, dataset):
#     z = float(dataset["lum_z"].iloc[galaxy])
#     r = float(dataset["lum_r"].iloc[galaxy])
#     g = float(dataset["lum_g"].iloc[galaxy])
#     m = float(dataset["mass"].iloc[galaxy])
#     rad = float(dataset["rad"].iloc[galaxy])
#     return [z, r, g, m, rad]


def Potential_depth(mass, radius):
    """Metric of the potential depth of the (central) galaxy - tells about its potential impact on satellites.
    Mass and radius are stellar mass and stellar half mass radius. Metric by Boryana's paper '10.1093/mnras/staa623'."""
    return 0 if radius == 0 else mass/radius


def Virialized_extent(v_disp_2, mass, radius):
    """Measure of how virialized halo is. Metric by Boryana's paper '10.1093/mnras/staa623'."""
    return mass/(radius*v_disp_2)


def Mass_proxy(v_disp2, distance, radius):
    """Metric by Boryana's paper '10.1093/mnras/staa623'."""
    # will be zero for central galaxy and is useless -> may be interesting only to add as an information about galaxies in the environment
    # We do not use this anyway because its not DESI observable
    mass_proxy1 = v_disp2*distance
    mass_proxy2 = v_disp2*radius
    return mass_proxy1, mass_proxy2


def Central_properties(c_galaxy, position, coordinates, dataset):
    """Get luminosities, stellar half mass rad, and stellar mass of the (central) galaxy."""
    z, r, g, m, rad = Galaxy_properties(c_galaxy, dataset)
    dist = np.linalg.norm(coordinates[c_galaxy]-position)
    v_disp2 = dataset["v_disp2"].iloc[c_galaxy]
    mass_proxy1, mass_proxy2 = Mass_proxy(v_disp2, dist, r)
    return [z, r, g, m, rad, Virialized_extent(v_disp2, m, r), Potential_depth(m, r), mass_proxy1, mass_proxy2]


def Galaxy_properties_2(galaxy, position, coordinates, dataset):
    """Calculate 'secondary, derived galaxy properties, such ass mass proxy and potential depth.'"""
    # not interested in (stellar) radius
    z, r, g, m, _ = Galaxy_properties(galaxy, dataset)
    dist = np.linalg.norm(coordinates[galaxy]-position)
    v_disp2 = dataset["v_disp2"].iloc[galaxy]
    T = v_disp2*m
    V = 0 if dist == 0 else m/dist
    mass_proxy = v_disp2*dist
    return [z, r, g, m, T, V, mass_proxy]


def Neighborhood_properties(position, c_galaxy, kd_tree, coordinates, dataset):
    """Retrun maximum value of each galaxy property for all neighbors within 5 Mpc/h (not used)."""
    R = 5_000
    neighbors_within = kd_tree.query_ball_point(position, R, p=2.0)
    # only central galaxy or central galaxy is even further away than 3Mpc/h
    if len(neighbors_within) <= 1:
        return Galaxy_properties_2(c_galaxy, position, coordinates, dataset)[3:]
    properties = []

    for neighbor in neighbors_within:
        # print(coordinates.shape)
        # print(neighbors_within)
        properties.append(Galaxy_properties_2(
            neighbor, position, coordinates, dataset))

    def iterate_preoprties_(properties, N):
        res = []
        for i in range(len(properties)):
            res.append(properties[i][N])
        return res
    return [np.sum(iterate_preoprties_(properties, i)) for i in range(3, 7)]


def Get_central_galaxy(galaxy, halo_pos, dataset, kd_tree, calculate_central=True):
    """Return the heaviest galaxy within 1 Mpc/h."""
    if calculate_central == False:
        return galaxy
    else:  # return nearest heaviset galaxy
        R = 1000  # kpc/h (in redshift space)
        nearest_neighbors = kd_tree.query_ball_point(halo_pos, R, p=2)
        masses = [dataset["mass"].iloc[i] for i in nearest_neighbors]
        if len(masses) == 0:  # Try finding in larger radius
            # no nearby gakaxy -> return "self" galaxy
            return galaxy
        else:  # return nearest heaviest
            return nearest_neighbors[masses.index(max(masses))]


def Distances_N(position, kd_tree, neighbors_list=[2]):
    """Find distanced to nearest galaxies."""
    distances, _ = kd_tree.query(position, neighbors_list, p=2)
    return list(distances)


def Get_galaxy_position(dataset, galaxy):
    """Get the position (whether its redshift or real-space position is determined before in the scripy 'samples.py')."""
    return [dataset[i].iloc[galaxy] for i in ["posX", "posY", "posZ"]]


def Get_halo_properties(dataset, galaxy):
    """Get the properties of the host halo of the given galaxy."""
    properties = ["halo_conc", "halo_conc_log", "halo_mass",
                  "halo_index", "crit200", "crit500", "mean200", "group_vel"]
    properties = [dataset[i].iloc[galaxy] for i in properties]
    return properties


def New_feature_names(n_distances=1, list_neighbors=[0.25, 0.5, 1, 2, 3, 4, 5]):
    """Assign feature names for final  dataset"""
    distances = [f"dist{i}" for i in range(n_distances)]
    N_neighbors = [f"neigh{i}" for i in list_neighbors]
    central_features = ["anisotropy", "lum_z", "lum_r", "lum_g", "mass", "rad",
                        "vir_ext", "pot_depth", "mass_proxy1", "mass_proxy2"]
    neighborhood_features = ["sum_m", "sum_T", "sum_V", "sum_mass_proxy"]
    galaxy_position = ["posX", "posY", "posZ"]
    halo_properties = ["halo_conc", "halo_conc_log", "halo_mass",
                       "halo_index", "crit200", "crit500", "mean200", "grouop_vel"]
    feature_names = ["ID"] + distances + N_neighbors + central_features + \
        neighborhood_features + galaxy_position + halo_properties
    features_to_standardize = distances + N_neighbors + central_features + \
        neighborhood_features  # to perform normalization / standardization on them
    return feature_names, features_to_standardize


def Calculate_features(galaxy, coordinates, kd_tree, dataset):
    """Calculate all features"""
    halo_pos = coordinates[galaxy]
    central_galaxy = Get_central_galaxy(galaxy, halo_pos, dataset, kd_tree)
    halo_pos = coordinates[central_galaxy]
    # Now we assume that halo positions is the same as central galaxy position (which might be the same as the original galaxy in the first place!)
    distances = Distances_N(halo_pos, kd_tree)
    neighbors_within = Neighbors_within_R(halo_pos, kd_tree)
    anisotropy_metric = Anisotropy(
        halo_pos, kd_tree, coordinates, 5_000)  # 5 Mpc/h
    central_features = Central_properties(
        central_galaxy, halo_pos, coordinates, dataset)
    neighborhood_features = Neighborhood_properties(
        halo_pos, central_galaxy, kd_tree, coordinates, dataset)
    return distances + neighbors_within + anisotropy_metric + central_features + neighborhood_features
