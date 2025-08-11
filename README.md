# GRL_CM1_paper_code_Berman


The code on this repo was used to generate the figures in Berman et al. (submitted) to GRL. 
Also included is the same CM1 namelist to generate the model runs and the modified default sounding. 

advanced_skewt._plot.ipynb provides the code to modify the default sounding in input_sounding_default as well as making Figures 1a and 1b. 

input_sounding_default is the modified, default sounding used in the paper and can be used to initialize a CM1 simulation

namelist.input.template is a sample namelist that can be used to generate a CM1 simulation and shows how to modify the 1/4 circle hodograph wind profile

needed_functions.py are the functions needed to do the tracking in the updated_automated_area.py

processed_sounding_data.ipynb is the methodology to modify the observed sounding for both input into CM1 and to change the lapse rates/static stability in the UTLS. 

results_figures.ipynb produces Figs 2 and 3. 

updated_automated_area.py does the updraft tracking as described in the text. 
