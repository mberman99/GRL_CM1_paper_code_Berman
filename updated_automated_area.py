if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from netCDF4 import Dataset
    from matplotlib import cm
    import xarray as xr
    import pandas as pd
    import datetime
    import warnings # Silence the warnings from SHARPpy
    warnings.filterwarnings("ignore")
    from scipy import interpolate
    from scipy.ndimage import gaussian_filter
    import scipy
    import math
    from metpy.interpolate import cross_section
    from shapely.geometry import Polygon, Point, MultiPoint, GeometryCollection
    import shapely.geometry as geometry
    from shapely import concave_hull
    import cmweather
    from skimage.filters import gaussian
    from skimage.segmentation import active_contour
    from skimage import measure
    from skimage.draw import ellipse, polygon, polygon_perimeter
    from skimage.measure import label, regionprops, regionprops_table
    from skimage.transform import rotate
    from skimage import data, io, segmentation, color
    from skimage.color import label2rgb
    import matplotlib.patches as mpatches
    from shapely.validation import make_valid
    import glob
    import metpy.calc as mpcalc
    from metpy.units import units 
    from sharppy.sharptab import winds, utils, params, thermo, interp, profile
    from sharppy.io.spc_decoder import SPCDecoder
    import sharppy.plot.skew as skew


    from matplotlib.ticker import ScalarFormatter, MultipleLocator
    from matplotlib.collections import LineCollection
    import matplotlib.transforms as transforms
    from metpy.plots import add_metpy_logo, Hodograph, SkewT
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import sharppy.sharptab as tab
    from matplotlib import gridspec
    from scipy.spatial.distance import cdist  

    #Import functions
    import needed_functions as imported_functions

    from dask.distributed import Client
    client = Client(memory_limit = '8GB')

    import gc
    gc.set_threshold(10000, 10, 10)

    def get_lapse_n2(init_condition_file, trop_method):

        th = init_condition_file.th[:,0,0].values*units.degree_Kelvin
        prs = init_condition_file.prs[:,0,0].values*units.pascal   
        qv = init_condition_file.qv[:,0,0].values*units.kg/units.kg
        hgt = init_condition_file.zh.values*units.kilometer
        hgt_m = hgt.to(units.meter)
        uint = init_condition_file.uinterp[:,0,0].values*units.meter_per_second
        vint = init_condition_file.vinterp[:,0,0].values*units.meter_per_second

        prs_hpa = prs.to(units.hectopascal)
        temp = mpcalc.temperature_from_potential_temperature(prs_hpa, th)
        tempc = temp.magnitude-273.15
        tempc = tempc*units.degC
        sh = mpcalc.specific_humidity_from_mixing_ratio(qv)
        dewp = mpcalc.dewpoint_from_specific_humidity(prs, tempc, sh)
        dewp = dewp
        #dewp[0:119] = dewp[0:119] - 2
        u_kts = uint.to(units.knot)
        v_kts = vint.to(units.knot)
        wnddir = mpcalc.wind_direction(u_kts, v_kts)
        wndspd = mpcalc.wind_speed(u_kts, v_kts)


        
        #Calculate the EL to do the calculations
        el_pressure, e_temperature = mpcalc.el(prs_hpa, tempc, dewp, which = 'bottom')
        parc_prof = mpcalc.parcel_profile(prs_hpa, tempc[0], dewp[0])
        cape = mpcalc.cape_cin(prs_hpa, tempc, dewp, parc_prof)[0]

        el_pres, el_idx = imported_functions.find_nearest(prs_hpa.magnitude, el_pressure.magnitude)
        el_hgt = hgt[el_idx]
        dz = [hgt_m.magnitude[ind+1] - hgt_m.magnitude[ind]for ind in range(len(hgt)-1)]

        trop_top_layer = el_pressure.magnitude - 50

        top_pres, top_idx = imported_functions.find_nearest(prs_hpa, trop_top_layer)
        pres_int = prs_hpa[el_idx:top_idx].magnitude

        #Brunt-Vaisala frequency from PTGT to 50 hPa above the tropopause
        dtdz = mpcalc.first_derivative(th.magnitude, delta = dz)
        n2_hand = 9.80665/th * dtdz  
        n2_handt = np.mean(n2_hand[el_idx:top_idx]).magnitude

        bvs = mpcalc.brunt_vaisala_frequency_squared(hgt_m, th)

        mean_n2 = np.mean(bvs[el_idx:top_idx])

        #Lapse Rate
        #SharpPy profile to get the lapse rates
        prof = profile.create_profile(profile='default', pres=prs_hpa, hght=hgt_m, tmpc=tempc, \
                                                dwpc=dewp, wspd = wndspd, wdir = wnddir, strictQC=False)
        lapses = []
        for a in pres_int:
            lapses.append(params.lapse_rate(prof, a, a+1))
            mean_lapse = np.mean(lapses)
        del lapses

        return el_hgt.magnitude, el_pressure.magnitude, mean_lapse, mean_n2.magnitude, cape.magnitude

    def get_masked_info(area_polygons, ti, z):
        finalMask =  xr.full_like(open_files.winterp[0,0,200:600,200:600], np.nan)
        y_up, x_up = area_polygons.exterior.coords.xy
        temp_mask = imported_functions.serial_mask(x_grid, y_grid, area_polygons)
        temp_mask = xr.DataArray(temp_mask, dims=['xh', 'yh'])
        temp_mask = (temp_mask.where(temp_mask)).fillna(0)
        finalMask = finalMask.fillna(0) + temp_mask
        finalMask = finalMask.where(finalMask > 0)

        current_file = open_files.isel(time = ti)
        current_file = current_file.sel(zh = z)

        masked_data = current_file['smoothed_velo'][200:600, 200:600].where(finalMask >= 1)
        max_velo = masked_data[:,:].max(skipna=True).values.item()

        return max_velo

    def get_uca_ot_labels(velo_field, velo_thresh):
        """ This function uses the skimage label tool to identify the 
        midlevel updraft core and the OT based on different velocity thresholds."""

        #Clean and mask the data
        velo_area = velo_field[:600, :600]
        mask_velo = velo_area > velo_thresh

        #Mid-level labels
        velo_labeled = label(mask_velo)
        regions_mid = regionprops(velo_labeled)
        props_mid = regionprops_table(velo_labeled,  properties = ('centroid', 'coords', 'area'))

        return props_mid, regions_mid
    def get_median_polygon_area(uc_info):
        """ This function gets the median value from the alpha shape function 
            to find the median polygon areas from the OT and UC polygons"""
        a_values = np.arange(0, 1.1, 0.1)
        uc_areas = np.zeros(len(a_values))

        n = 0
        for a_value in a_values:
            ucar = imported_functions.alpha_shape(uc_info['coords'], alpha=a_value)
            uc_areas[n] = ucar[0].area
            #low_areas[n] = lowr.area 
            n+=1

        uc_med_area_out = np.median(uc_areas)
        med_uc_idx = np.where(uc_areas == uc_med_area_out)

        if len(med_uc_idx[0]) > 1:
            med_idx_uc = med_uc_idx[0][0]
        else:
            med_idx_uc = med_uc_idx

        alpha_med_uc = a_values[med_idx_uc]

        return uc_med_area_out, alpha_med_uc
    lower_shear = [4]
    upper_shear = [29]
    print("Now starting the for loop")
    for hl1 in lower_shear:
        for hu1 in upper_shear:
                hash = 'hodo_test_'+str(hl1)+'_'+str(hu1)+'_1m_dss_full_'
                print(hash)

                wrf_base='/data/keeling/a/melinda3/NASA/cm1_radiation/temp_cm1_dir/'

                path_to_files=wrf_base+hash

                open_files = xr.open_mfdataset(path_to_files+'/cm1out_0*.nc')

                el_z, el_pres, lapse_el, n2_el, cape_init = get_lapse_n2(open_files.sel(time = '00:00:00'),\
                "PTGT")

                uca_out = np.zeros(len(open_files.time))
                ota_out = np.zeros(len(open_files.time))
                low_out = np.zeros(len(open_files.time))
                updraft_top = np.zeros(len(open_files.time))
                updraft_bot = np.zeros(len(open_files.time))
                srh_out = np.zeros(len(open_files.time))
                bshear_out = np.zeros(len(open_files.time))
                ot_df_final_out = []
                uc_df_final_out = []
                w_max_trop = np.zeros(len(open_files.time))
                lapse_out = np.zeros(len(open_files.time))
                n2_out = np.zeros(len(open_files.time))
                w_mid = np.zeros(len(open_files.time))
                w_mid_avg = np.zeros(len(open_files.time))
                finalMask = xr.full_like(open_files.winterp[0,0,200:600,200:600], np.nan)
                coreMask = xr.full_like(open_files.winterp[0,0,200:600,200:600], np.nan)
                lowMask = xr.full_like(open_files.winterp[0,0,200:600,200:600], np.nan)
                trop_temp = 216.25
                trop_height = el_z.item()
                core_height = 6
                low_height = 4.625
            
                x_grid, y_grid = np.meshgrid(np.arange(len(open_files.xh[200:600])), np.arange(len(open_files.yh[200:600])))
                slice_low_smoothed = gaussian_filter(open_files['winterp'], sigma = 2)
                xr3 = xr.DataArray(slice_low_smoothed, dims = ['time', 'zh', 'xh', 'yh'])
                open_files['smoothed_velo'] = xr3
                t = np.where(open_files.zh.values < 17.5)
                height_vals = open_files.zh.values[t]
                core_velo = 20
                trop_velo = 20
                polygon_dfs = pd.DataFrame()
                polygon_dfs = pd.DataFrame()
                for ti in np.arange(0, 100):
                    for z in height_vals:
                        current_file = open_files.isel(time = ti)
                        if ti == 0:
                            sliced_file = current_file.sel(zh = slice(0, 6.5))
                            p = sliced_file.prs[:,0,0].values *100 *units.hectopascal
                            test_z = sliced_file.zh.values*units.kilometer
                            u = sliced_file.uinterp[:,0,0].values * units.meter_per_second
                            v = sliced_file.vinterp[:,0,0].values * units.meter_per_second
                            depth = 3 * units.kilometer
                            srh_init = mpcalc.storm_relative_helicity(test_z, u, v, depth)[2].magnitude
                            ubshr3, vbshr3 = mpcalc.bulk_shear(p, u, v, height=test_z, depth=depth)
                            bshear_init = mpcalc.wind_speed(ubshr3, vbshr3).magnitude

                            d2 = 6 * units.kilometer
                            ubshr6, vbshr6 = mpcalc.bulk_shear(p, u, v, height=test_z, depth=d2)
                            bshear6 = mpcalc.wind_speed(ubshr6, vbshr6).magnitude
                            cape = cape_init
                            
                        current_file = current_file.sel(zh = z)
                        if current_file.time.values.astype("timedelta64[m]") < 45:
                            
                            max_uca_test = imported_functions.get_uc_center(current_file['smoothed_velo'][200:600, 200:600], core_velo)
                            pm, rm = get_uca_ot_labels(current_file['smoothed_velo'], core_velo)
                            pm_df = pd.DataFrame(pm)
                            pm_df['time'] = ti
                            pm_df['z'] = z 

                            upd_info = imported_functions.ot_mid_loc_area(pm_df, max_uca_test)
                            uca_x = upd_info['centroid-1']
                            uca_y = upd_info['centroid-0']
                            polygon_dfs = pd.concat([polygon_dfs, pm_df])

                        if 45 <= current_file.time.values.astype("timedelta64[m]") <= 47:
                            pm, rm = get_uca_ot_labels(current_file['smoothed_velo'][200:600, 200:600], core_velo)
                            pm_df = pd.DataFrame(pm)
                            pm_df['time'] = ti
                            pm_df['z'] = z 
                            if ((((pm_df.area > 25).any()) == True)):
                                pm_df = pm_df[pm_df.area > 25]
                                upd_info = imported_functions.find_new_uc_update(pm_df, uca_x, uca_y)
                                uca_x = upd_info['centroid-1']
                                uca_y = upd_info['centroid-0']
                                polygon_dfs = pd.concat([polygon_dfs, pm_df])

                        if current_file.time.values.astype("timedelta64[m]") >= 48:
                            pm, rm = get_uca_ot_labels(current_file['smoothed_velo'][200:600, 200:600], core_velo)
                            pm_df = pd.DataFrame(pm)
                            pm_df['time'] = ti
                            pm_df['z'] = z 
                            pm_df['srh03'] = srh_init
                            pm_df['bshear06'] = bshear6
                            pm_df['lapse_rate'] = lapse_el
                            pm_df['n2'] = n2_el
                            polygon_dfs = pd.concat([polygon_dfs, pm_df])
             


                #Using the last known polygon centroid, track the mid polygons and return a pandas df with only the selected ones
                tracked_mids = pd.DataFrame()
                for ti in np.arange(45,100):
                    for z in height_vals:
                        pm_df = polygon_dfs[(polygon_dfs.time == ti) & (polygon_dfs.z == z)]
                        if ((((pm_df.area > 25).any()) == True)):
                            pm_df = pm_df[pm_df.area > 25]
                            dists = (uca_y - pm_df['centroid-0']) ** 2 + (uca_x - pm_df['centroid-1']) ** 2
                            dists = dists[dists < 250]
                            if len(dists) == 0:
                                upd_info = upd_info[:1]
                                upd_info['polygon_med_area'] = np.nan
                                upd_info['median_alpha'] = np.nan
                                uca_x = uca_x
                                uca_y = uca_y
                                upd_info['conv_hull'] = conv_up
                                upd_info['srh03'] = srh_init
                                upd_info['bshear06'] = bshear6
                                upd_info['lapse_rate'] = lapse_el
                                upd_info['n2'] = n2_el
                            else:
                                upd_info = pm_df.iloc[dists.argsort()]
                                upd_info = upd_info[:1]
                                upd_info['polygon_med_area'] = get_median_polygon_area(upd_info.iloc[0])[0]
                                alpha_out = get_median_polygon_area(upd_info.iloc[0])[1]
                                upd_info['median_alpha'] = alpha_out
                                uca_x = upd_info['centroid-1'].iloc[0]
                                uca_y = upd_info['centroid-0'].iloc[0]

                                conv_up, _ = imported_functions.alpha_shape(upd_info['coords'].iloc[0], alpha_out)
                                if conv_up.type == 'MultiPolygon':
                                    conv_up = max(conv_up.geoms, key=lambda a: a.area)
                                    upd_info['polygon_med_area'] = conv_up.area 

                                upd_info['conv_hull'] = conv_up
                                upd_info['srh03'] = srh_init
                                upd_info['bshear06'] = bshear6
                                upd_info['lapse_rate'] = lapse_el
                                upd_info['n2'] = n2_el

                            tracked_mids = pd.concat([tracked_mids, upd_info[:1]])

 

                t3= list(map(get_masked_info, tracked_mids['conv_hull'], tracked_mids['time'], tracked_mids['z']))

                tracked_mids['max_velo'] = t3
                #tracked_mids['max_th'] = t4
                #tracked_mids['max_qv'] = t5


                tracked_mids.to_csv('/data/keeling/a/melinda3/gpm/data_dir/final_tracking_algo_'+hash+'.csv')

                        
        
    
