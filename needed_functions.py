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

from metpy.units import units

def get_uc_center(data, w_core):
    """ This function determines the area where the updraft core is
    strongest for a storm. This setting has it examining the right moving storm exclusively
    but it can be modified to look at the left mover or both storms. It also 
    can look at the storms pre-split."""
    #Wcore = 20 m/s, core_level = 6.25 km
    w = data[:600, :600]
    uc = w.where(w.values > w_core)
    max_uca = np.where(uc == uc.max())
    return max_uca

    
#Search the nearest 20 grid cells to the UCA for the max velocity to get the location of the OT
def find_ot_from_uca(velo, max_uca_location, search_radius):
    uc_center_x = max_uca_location[0][0]
    uc_center_y = max_uca_location[1][0]
    

    min_x = uc_center_x - search_radius
    max_x = uc_center_x + search_radius

    min_y = uc_center_y - search_radius
    max_y = uc_center_y + search_radius


    btd = np.max(velo[min_x:max_x, min_y:max_y]).values
    btd_loc = np.where(velo == btd)
    return btd, btd_loc
    

#Determine the midlevel UCA and OT polygons based on the velocities at 6.25 km and right above the tropopause
def get_uca_ot_labels(cm1_lower, cm1_upper, cm1_lowest, velo_thresh_mid, velo_thresh_upper, time):
    """ This function uses the skimage label tool to identify the 
    midlevel updraft core and the OT based on different velocity thresholds."""

    #Clean and mask the data
    mid_level_velo = cm1_lower[time, :600, :600]
    upper_velo = cm1_upper[time, :600 ,: 600]
    low_velo = cm1_lowest[time, :600, :600]
    mask_upper = upper_velo > velo_thresh_upper
    mask_mid = mid_level_velo > velo_thresh_mid
    mask_lower = low_velo > velo_thresh_mid

    #Mid-level labels
    label_mid = label(mask_mid)
    regions_mid = regionprops(label_mid)
    props_mid = regionprops_table(label_mid,  properties = ('centroid', 'coords'))

    #OT labels
    label_ot = label(mask_upper)
    regions = regionprops(label_ot)
    props_ot = regionprops_table(label_ot, properties = ('centroid', 'coords'))

    #Low-level labels
    label_low = label(mask_lower)
    regions_low = regionprops(label_low)
    props_low = regionprops_table(label_low, properties = ('centroid', 'coords'))

    return props_mid, props_ot, props_low, regions_mid, regions, regions_low

#Get the mid level UC location, UCA, OT location and OTA
def ot_mid_loc_area(df_mid, max_uc_loc):
    """ This function gets the location of the midlevel UC and OT
    from the lavels in get_uca_ot_labels and also reports the area for them."""

    # 1. Find the location of the midelevel updraft based on the location of the place of max velo at the midlevels
    input_y = max_uc_loc[1]
    input_x = max_uc_loc[0]
    mid_level_y = df_mid.iloc[(df_mid['centroid-1']-input_x).abs().argsort()]
    mid_level_uca = mid_level_y.iloc[(mid_level_y['centroid-0']-input_y).abs().argsort()[:1]]

    return mid_level_uca

def serial_mask(lon, lat, polygon):
    """Masks longitude and latitude by the input shapefile.
    Args:
        lon, lat: longitude and latitude grids.
            (use np.meshgrid if they start as 1D grids)
        polygon: output from `select_shape`. a shapely polygon of the region
                you want to mask.
    Returns:
        mask: boolean mask with same dimensions as input grids.
    Resource:
    adapted from https://stackoverflow.com/questions/47781496/
                    python-using-polygons-to-create-a-mask-on-a-given-2d-grid

    From: https://gist.github.com/bradyrx/1a15d8c45eac126e78d84af3f123ffdb
    """
    # You might need to change this...
    if ( (len(lon.shape) != 2) | (len(lat.shape) != 2) ):
        raise ValueError("""Please input a longitude and latitude *grid*.
            I.e., it should be of two dimensions.""")
    lon, lat = np.asarray(lon), np.asarray(lat)
    lon1d, lat1d = lon.reshape(-1), lat.reshape(-1)
    # create list of all points in longitude and latitude.
    a = np.array([Point(y, x) for x, y in zip(lon1d, lat1d)], dtype=object)
    # loop through and check whether each point is inside polygon.
    mask = np.array([polygon.contains(point) for point in a])
    # reshape to input grid.
    mask = mask.reshape(lon.shape)
    return mask

def get_updraft_top_bottom(masked_data_low):
    levs_idx_low = []
    levs_idx_high = []
    for lev in np.arange(len(masked_data_low.zh)):
        lev_data_low = masked_data_low[lev, :, :]

        if np.any(lev_data_low > 20):
            levs_idx_low.append(1)
        else:
            levs_idx_low.append(0)

    ar_levs_low = np.array(levs_idx_low)

    met_crit_low = np.where(ar_levs_low == 1)

    for lev in np.arange(len(masked_data_high.zh)):
        lev_data_high = masked_data_high[lev, :, :]

        if np.any(lev_data_high > 20):
            levs_idx_high.append(1)
        else:
            levs_idx_high.append(0)

    ar_levs_high = np.array(levs_idx_high)

    met_crit_high = np.where(ar_levs_high == 1)

    bot = met_crit_low[0][0]
    top = met_crit_high[0][-1]

    return top, bot

def get_avg_core_updraft(masked_data_mid):
    w_mid_avg = []
    w_mid = np.zeros(len(masked_data_mid.zh))

    for z in np.arange(0, len(masked_data_mid.zh)):
        mdm = masked_data_mid[z, :, :].max(skipna=True)
        w_mid[z] = mdm.values.item()
        mdm = 0
    w_mid_avg = np.mean(w_mid)
    w_mid_avg = w_mid_avg
    w_mid = []
    
    return w_mid_avg

    #Using the coordinates from the OT df, calculate the area using Green's theorem
def get_polygons(ot_df, n):
        """ This function gets the corrected OT area polygon from the coordinates from the labelled shapes from 
            the ID step and skimage. Using shapely, the polygon exteriors are constructed. Can also get the
            area from the polygons""" 
        ot_chosen = ot_df['coords'].iloc[n]
        ot1_x = ot_chosen[:,1]
        ot1_y = ot_chosen[:,0]
        first_coord = []
        second_coord = []
        for y_val_ux in np.unique(ot1_y):
            ys = ot_chosen[np.where(ot_chosen[:,0] == y_val_ux)]
            first_coord.append(ys[0])
            second_coord.append(ys[-1])

        second_coord = np.flip(second_coord, axis = 0)
        sorted_dims = np.concatenate((first_coord, second_coord))
        ot1_x = sorted_dims[:,1]
        ot1_y = sorted_dims[:,0]
        poly1 = Polygon(zip(ot1_x, ot1_y))

        ota = poly1.area

        return ota, poly1

def modify_poly(ot_polygon):
        otx, oty = ot_polygon.exterior.xy
        xflip = np.flip(otx)
        xfr = xflip.reshape((-1,1))
        yflip = np.flip(oty)
        yfr = yflip.reshape((-1,1))
        ot_coords = np.hstack([xfr, yfr])

        return ot_coords, xfr, yfr

def area(vs):
        a = 0
        x0,y0 = vs[0]
        for [x1,y1] in vs[1:]:
            dx = x1-x0
            dy = y1-y0
            a += 0.5*(y0*dx - x0*dy)
            x0 = x1
            y0 = y1
        return a

## Test the alpha shape method from 
# https://gist.github.com/jclosure/d93f39a6c7b1f24f8b92252800182889

from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import numpy as np
import math

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. Smaller
                numbers don't fall inward as much as larger numbers. Too large,
                and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense in computing an alpha
        # shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
    
    coords = np.array([point for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def get_median_polygon_area(uc_info):
    """ This function gets the median value from the alpha shape function 
        to find the median polygon areas from the OT and UC polygons"""
    a_values = np.arange(0, 1.1, 0.1)
    uc_areas = np.zeros(len(a_values))

    n = 0
    for a_value in a_values:
        ucar = alpha_shape(uc_info['coords'][0], alpha=a_value)
        uc_areas[n] = ucar.area
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


def find_new_uc_update(uc_polygons, uca_x_prev, uca_y_prev):
    """This function finds the UCA polygon of interest after the first time step 
    of analysis and only searches within 5 grid cells (2.5km) to find the next core. 
    This is to prevent updrafts that have a larger max w at 6.25 km 
    from skewing the time series analysis. """

    # 1. Find the location of the midelevel updraft based on the location of the place of max velo at the midlevels
    mid_level_y = uc_polygons.iloc[(uc_polygons['centroid-0']-uca_y_prev).abs().argsort()]
    mid_level_uca = mid_level_y.iloc[(mid_level_y['centroid-1']-uca_x_prev).abs().argsort()[:1]]

    return mid_level_uca

def get_closest_ot_to_uc(ucx, ucy, po_df):
        import pandas as pd
        df_test = pd.DataFrame()
        df_test['point'] = [(ucx.item(), ucy.item())]

        df_test2 = pd.DataFrame()
        po_df['point'] = [(x, y) for x,y in zip(po_df['centroid-1'], po_df['centroid-0'])]

        df_test['closest'] = [closest_point(x, list(po_df['point'])) for x in df_test['point']]

        coord_1 = df_test['closest'].item()[0]

        matching_ot = po_df[po_df['centroid-1'] == coord_1]

        return matching_ot
from scipy.spatial.distance import cdist  
      
def closest_point(point, points):
        """ Find closest point from a list of points. """
        return points[cdist([point], points).argmin()]


#Create a function that uses the initial conditions to take calculate lapse rates and N^2

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

    el_pres, el_idx = find_nearest(prs_hpa.magnitude, el_pressure.magnitude)
    el_hgt = hgt[el_idx]
    dz = [hgt_m.magnitude[ind+1] - hgt_m.magnitude[ind]for ind in range(len(hgt)-1)]

    trop_top_layer = el_pressure.magnitude - 50

    top_pres, top_idx = find_nearest(prs_hpa, trop_top_layer)
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

def find_nearest(array, value):
    array = np.asarray(array)
    step = (np.abs(array - value)) 
    idx = np.nanargmin(step)
    return array[idx], idx

def proper_round(num, dec=0):
    "From https://stackoverflow.com/questions/31818050/round-number-to-nearest-integer"
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
    return float(num[:-1])