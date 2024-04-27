import itertools


def _transform(word):
    """
    Feature name transformations for nicer plots
    """
    if word == "Vol":
        return r"$V_\mathrm{m}$"
    elif word == "Melt":
        return r"$T_\mathrm{m}$"
    elif word == "M":
        return r"$M$"
    elif word == "Atomic":
        return r"$Z$"
    elif word == "Electro":
        return r"$\mathrm{EN}$"
    elif word == "%ic":
        return r"$\%\mathrm{IC}$"
    elif word == "Bulk_m":
        return r"$K$"
    elif word == "Shear_m":
        return r"$G$"
    elif word == "Coh_en":
        return r"$E_{\mathrm{coh}}$"
    elif word == "Form_e_per_atom":
        return r"$H$"
    elif word == "Group":
        return r"$N_{\mathrm{G}}$"
    elif word == "Row":
        return r"$N_{\mathrm{P}}$"
    elif word == "Density":
        return r"$\rho$"
    elif word == "Radius":
        return r"$R$"
    elif word == "Melt_temp_K":
        return r"$T_{\mathrm{m}}\ (K)$"
    elif word == "Melt_temp_C":
        return r"$T_{\mathrm{m}}\ (C)$"
    elif word == "Log10_melt_temp_K":
        return r"$T_{\mathrm{m}}\ (log10, K)$"
    elif word == "Log10_melt_temp_C":
        return r"$T_{\mathrm{m}}\ (log10, C)$"
    elif word == "D_T_Vegard":
        return r"$T_{\mathrm{m}}\ (Vegard)$"
    else:
        return word


def rename_features(features, averaging=None):
    """Feature name transformations for nicer plots

    Parameters
    ----------
    features: dict or list
        list or dictionary of features
    averaging: str
        key to features dict of features used (default is None)

    Returns
    -------
    list
        new names of features in features[averaging!=None]
    """
    new_names = []
    if averaging:
        features = features[averaging]
    for name in features:
        name = name[0].upper() + name[1:]
        if name[-4:] == "harm":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{h}}$")
        elif name[-5:] == "harmX":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{hw}}$")
        elif name[-4:] == "quad":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{q}}$")
        elif name[-5:] == "quadX":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{qw}}$")
        elif name[-3:] == "ave":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{a}}$")
        elif name[-4:] == "aveX":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{aw}}$")
        elif name[-2:] == "sd":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{s}}$")
        elif name[-3:] == "sdX":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{sw}}$")
        elif name[-3:] == "geo":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{g}}$")
        elif name[-4:] == "geoX":
            name = _transform(name.split("_")[0])
            new_names.append(r"$\langle$" + name + r"$\rangle$" + r"$_{\mathrm{gw}}$")
        else:
            name = _transform(name)
            new_names.append(name)
    return new_names


def construct_features_dictionary():
    """Constructing different combinations of features

    Returns
    -------
    dict
        dicitonary of different combinations of features
    """
    features_all = [
        "coh_en",
        "bulk_m",
        "shear_m",
        "form_e_per_atom",
        "density",
        "%ic",
        "M_ave",
        "M_aveX",
        "M_sd",
        "M_sdX",
        "M_harm",
        "M_harmX",
        "M_geo",
        "M_geoX",
        "M_quad",
        "M_quadX",
        "radius_ave",
        "radius_aveX",
        "radius_sd",
        "radius_sdX",
        "radius_harm",
        "radius_harmX",
        "radius_geo",
        "radius_geoX",
        "radius_quad",
        "radius_quadX",
        "atomic_num_ave",
        "atomic_num_aveX",
        "atomic_num_sd",
        "atomic_num_sdX",
        "atomic_num_harm",
        "atomic_num_harmX",
        "atomic_num_geo",
        "atomic_num_geoX",
        "atomic_num_quad",
        "atomic_num_quadX",
        "row_ave",
        "row_aveX",
        "row_sd",
        "row_sdX",
        "row_harm",
        "row_harmX",
        "row_geo",
        "row_geoX",
        "row_quad",
        "row_quadX",
        "group_ave",
        "group_aveX",
        "group_sd",
        "group_sdX",
        "group_harm",
        "group_harmX",
        "group_geo",
        "group_geoX",
        "group_quad",
        "group_quadX",
        "electro_ave",
        "electro_aveX",
        "electro_sd",
        "electro_sdX",
        "electro_harm",
        "electro_harmX",
        "electro_geo",
        "electro_geoX",
        "electro_quad",
        "electro_quadX",
        "vol_ave",
        "vol_aveX",
        "vol_sd",
        "vol_sdX",
        "vol_harm",
        "vol_harmX",
        "vol_geo",
        "vol_geoX",
        "vol_quad",
        "vol_quadX",
        "melt_temp_ave",
        "melt_temp_aveX",
        "melt_temp_sd",
        "melt_temp_sdX",
        "melt_temp_harm",
        "melt_temp_harmX",
        "melt_temp_geo",
        "melt_temp_geoX",
        "melt_temp_quad",
        "melt_temp_quadX",
    ]

    # different sets of features
    ave = [f for f in features_all if "ave" in f]
    sd = [f for f in features_all if "sd" in f]
    harm = [f for f in features_all if "harm" in f]
    geo = [f for f in features_all if "geo" in f]
    quad = [f for f in features_all if "quad" in f]
    averages = [ave, sd, harm, geo, quad]

    features = {}

    features["all"] = features_all
    features["dft"] = [
        "coh_en",
        "bulk_m",
        "shear_m",
        "form_e_per_atom",
        "density",
        "%ic",
    ]

    # constructing features that are a combination of dft/compound features and only one type of statistical averaging of elemental features
    for average in averages:
        average1 = average[0].split("_")[-1]
        features[average1] = average + features["dft"]

    # constructing features that are a combination of dft/compound features and only two types of statistical averaging of elemental features
    # note that we did not filter out for repeating feature sets
    for i, j in list(itertools.combinations(averages, r=2)):
        average1 = i[0].split("_")[-1]
        average2 = j[0].split("_")[-1]
        features_list = list(set(i + j))
        features[average1 + "_" + average2] = features_list + features["dft"]

    return features
