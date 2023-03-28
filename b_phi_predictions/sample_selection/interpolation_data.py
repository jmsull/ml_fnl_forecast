# INTERPOLATION FROM LAZEYRAS, fig 1

# @ARTICLE{2023JCAP...01..023L,
#        author = {{Lazeyras}, Titouan and {Barreira}, Alexandre and {Schmidt}, Fabian and {Desjacques}, Vincent},
#         title = "{Assembly bias in the local PNG halo bias and its implication for f $_{NL}$ constraints}",
#       journal = {\jcap},
#      keywords = {cosmological parameters from LSS, galaxy clustering, non-gaussianity, cosmological simulations, Astrophysics - Cosmology and Nongalactic Astrophysics},
#          year = 2023,
#         month = jan,
#        volume = {2023},
#        number = {1},
#           eid = {023},
#         pages = {023},
#           doi = {10.1088/1475-7516/2023/01/023},
# archivePrefix = {arXiv},
#        eprint = {2209.07251},
#  primaryClass = {astro-ph.CO},
#        adsurl = {https://ui.adsabs.harvard.edu/abs/2023JCAP...01..023L},
#       adsnote = {Provided by the SAO/NASA Astrophysics Data System}
# }

INTERPOLATED_N = 86150

mass_bins = [11.79, 12.54, 13.28, 14.03]
# b_phi
unit_b = 10/1.9
b_low = [0-0.65*unit_b, 0 - 0.4*unit_b, 0 + 0.25*unit_b, 10 - 0.5*unit_b]
b_medium = [0-0.1*unit_b, 0 + 0.1*unit_b, 0 + 0.7*unit_b, 10 - 0.25*unit_b]
b_high = [10 - 0.8*unit_b, 10 - 0.6 * unit_b,
          10 - 0.4*unit_b, 10 + 0.75*unit_b]

# b_1
unit_b1 = 2.5/3.3
mean1, mean2, mean3, mean4 = 1.5*unit_b1, 2 * \
    unit_b1, 2.5-0.15*unit_b1, 2.5 + 1.95*unit_b1
b_low_1 = [mean1 + 0.2*unit_b1, mean2 + 0.3*unit_b1,
           mean3 + 0.55*unit_b1, mean4 + 0.7*unit_b1]
b_medium_1 = [mean1 - 0.05*unit_b1, mean2 - 0.1 *
              unit_b1, mean3 - 0.05*unit_b1, mean4 - 0.3*unit_b1]
b_high_1 = [mean1 - 0.1*unit_b1, mean2 - 0.25*unit_b1,
            mean3 - 0.45*unit_b1, mean4 - 0.2 * unit_b1]
