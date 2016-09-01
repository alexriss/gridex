"""
Grid-ex

Grid Spectroscopy in an IPython environment

Class to read Nanonis data files (grid spectroscopy).
The class can read binary data (.3ds) or a list of ascii files (*.dat)


2015, Alex Riss, GPL


Populates the dictionaries:
    self.headers        (headers in the files, keys are lowercase, space replaced by '_')
    
    and for each point in self.points:
        data_header    (headers for the particular point, such as x,y,z position)
        data           (np.array with the data)


        
Caveats:
    - binary values are read in in Big Endian format. I am not sure if that will change depending on the machine the Nanonis runs on
    - for ascii data square grids are assumed.
    
Todo:
    - check if the rotation calculation uses the right convention (clockwise vs anticlockwise) for ASCII data
    - better documentation (i.e. exmaples with better comments in the ipython example notebook), describe which data is generated
    - make smoothing and derivative options, such as in the nanonis viewers
    - have an interface to find images (based on date (+-3 days within the grid), based on other parameters (e.g. const height image, >4 Hz range in frequency shift))
        - plot STM/AFM images next to the grid
    - provide easy way to plot sweep channel at specific point, e.g. dI/dV maps from point spectroscopy grids
    - unit tests
Todo maybe:
    - switch image interpolation between 'nearest', 'bilinear', 'bicubic'
    - plot grid data as a function of the x,y values, not just the point number (sort of is done for binary files)
    - more interactive experience, either with matplotlib widgets, or with mpld3, or with bokeh

"""



from __future__ import print_function
from __future__ import division
import sys
import glob
import struct
import copy
import numpy as np
import numpy.lib.recfunctions as recfunctions  # seems we need to import this explicitely
import scipy.special
import scipy.optimize
import io
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets

import seaborn as sns
sns.set_style("dark")
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 1.5})
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})


MASS_ELECTRON = 9.10938356e-31    # kg
PLANCK_CONSTANT = 6.62607004e-34  # m2 kg / s
CONV_J_EV = 6.242e18             # conversion of Joule to eV

PYTHON_VERSION = sys.version_info.major

class GridData(object):
    """Class to read grid date from binary or ascii files. Also fits the KPFM parabola and the exponential IZ curves. """
    
    def __init__(self):
        self.filename = ""         # filename
        self.headers = {}          # dictionary containing all the header data
        self.data_points = []      # list holding the points, each point is a dictionary with the keys 'data_header' (containing header info from binary data) and 'data' (a structured numpy array containing the data from the sweeps)
        
        self.num_data_points = 0   # number of data points
        self.channels = []         # list holding the names of the swept channels
        self.points = 0            # number of points per sweep
        self.data = None           # 3-dimensional data as numpy array (first index is the number of the point, second index represents the channel, third index the index of the sweep)

        
    def load_spectroscopy(self,fname,long_output=False,first_file=None,last_file=None):
        """loads spectrscopy data, either the binary .3ds file or data from a number of .dat files. The optional parameters "first_file" and "last_file" can be used to further restrict the range for .dat files (substrings of the filename will work for those)."""
        ext = fname.rsplit(".",1)[-1]
        if  ext == "3ds":
            self.__init__()
            self._load_spectroscopy_3ds(fname,long_output)
        elif ext == "dat":
            self.__init__()
            self._load_spectroscopy_dat(fname,long_output,first_file,last_file)
        else:
            print("Error: Unknown file type \"%s\"." % ext)
            return False
        self.filename = fname
        
        
    def _load_spectroscopy_3ds(self,fname,long_output):
        """load spectroscopy data from a .3ds file"""
        
        if PYTHON_VERSION>=3:
            f = open(fname, encoding='utf-8', errors='ignore')
        else:
            f = open(fname)
        line = f.readline()
        while line:
            linestr = line.strip()
            if linestr.upper() == ':HEADER_END:': break
            line = f.readline()
            if not '=' in linestr: continue   # some line that does not contain a key-value pair
            
            key, val = linestr.split('=',1)
            key, val = key.strip(), val.strip()
            key = self._string_simplify1(key)
            key_ = self._string_simplify2(key)
            val = val.replace('"', '')
            
            # sometimes keys are used more than once (probably not in binary files, but I have seen it in the .dat files), just append an underline in this case
            while key in self.headers: key = key + "_"

            if key_ == 'grid_dim':
                vals = val.split('x')
                self.headers[key] = [int(x) for x in vals]
            elif key_ =='grid_settings':
                vals = val.split(';')
                self.headers[key] = [float(x) for x in vals]
            elif key_ in ['fixed_parameters', 'experiment_parameters', 'channels']:
                val = self._string_simplify1(val)
                vals = val.split(';')
                self.headers[key] = vals
            elif key_ in ['sweep_signal']:
                val = self._string_simplify1(val)
                self.headers[key] = val
            elif key_ == '#_parameters_(4_byte)':
                self.headers['num_parameters'] = self.headers[key] = int(val)
            elif key_ == 'experiment_size_(bytes)':
                self.headers['experiment_size'] = self.headers[key] = int(val)
            elif key_ == 'points':
                self.headers[key] = int(val)
            elif key_ in ['current_(a)','input_4_(v)','lix_1_omega_(a)','liy_1_omega_(a)','delay_before_measuring_(s)']:
                self.headers[key] = float(val)
            else:
                self.headers[key] = val
        
        self.num_data_points = self.headers['num_data_points'] = self.headers['grid_dim'][0] * self.headers['grid_dim'][1]
        self.channels = self.headers['channels']
        self.points   = self.headers['points']
        
        f.close()
        
        f = open(fname, 'rb')
        bindata = f.read()
        f.close()
        
        bindata = bindata[bindata.index(b':HEADER_END:')+len(':HEADER_END:')+2:]  # "+2+ is for the newline
        
        exp_size = self.headers['experiment_size'] + self.headers['num_parameters']*4;
        for i_point in range(self.num_data_points):
            offset_point = exp_size * i_point
            data_list = []        # list of the data, will be converted to structured numpy array at the end
            data_list_names = []  # the corresponding names
            data_headers = {}
            for i, par in enumerate(self.headers['fixed_parameters']+self.headers['experiment_parameters']):
                par_key = self._string_simplify1(par)
                while par_key in data_headers: par_key = par_key + "_"  # in case some names are used twice
                data_headers[par_key] = struct.unpack('>f',bindata[offset_point + i*4:offset_point + i*4 + 4])   # read header values in big endian mode
            for i_channel,channel in enumerate(self.headers['channels']):
                    
                offset_binheaders = self.headers['num_parameters'] * 4
                channel_start, channel_end = offset_point + offset_binheaders + i_channel*self.headers['points']*4, offset_point + offset_binheaders + (i_channel+1)*self.headers['points']*4
                
                #arr = np.fromfile(bindata[channel_start:channel_end], dtype = '>f4')  # dont know why this did not work, I think f4 for numpy might be different than f4 for struct, but didn't test
                arr = struct.unpack('>%sf' % self.points, bindata[channel_start:channel_end])
                data_list.append(arr)
                #data_list_names.append(channel.replace(' ','_').lower())
                data_list_names.append(channel)

            # created structured numpy array
            data_list_formats = ['float'] * len(data_list_names)
            data = np.array(data_list)
            data = np.core.records.fromarrays(data, names=data_list_names, formats=data_list_formats)
            #data = data.transpose()
            
            self.data_points.append({'data_headers': data_headers, 'data': data})
            
        # generate 3D numpy array for faster access and slicing
        self.data = self._generate_3D_data(self.data_points)
        
        
    def _load_spectroscopy_dat(self,fnames,long_output,first_file=None,last_file=None):
        """load spectroscopy data from a .dat files, the argument fname should be a unix-style list of files, i.e. "spectroscopy_4_*.dat"."""
        
        read_somedata = False
        if first_file:
            first_file_found=False
        files_read = 0
    
        for i_file,fname in enumerate(glob.glob(fnames)):
            # first_first and last_file limits
            if first_file and not first_file_found:
                if first_file in fname:
                    first_file_found=True
                else:
                    continue
            if last_file:
                if last_file in fname:
                    break
                
            if long_output: print("loading %s" % fname)
            f = open(fname)
            
            data = {}
            data_headers = {}
            
            line = f.readline()
            while line:
                linestr = line.strip()
                if linestr.upper() == '[DATA]': break
                line  = f.readline()
                if not '\t' in linestr: continue   # some line that does not contain a key-value pair
                
                key, val = linestr.split('\t',1)
                key, val = key.strip(), val.strip()
                key = self._string_simplify1(key)
                key_ = self._string_simplify2(key)
                val = val.replace('"', '')
                
                # sometimes keys are used more than once, just append an underline in this case
                while key in data_headers: key = key + "_"

                if key_ in ['x_(m)', 'y_(m)', 'z_(m)','current_(a)','final_Z_(m)','input_4_(v)','lix_1_omega_(a)','liy_1_omega_(a)']:
                    data_headers[key] = float(val)
                else:
                    data_headers[key] = val
                
                read_somedata = True
            
            names = f.readline().strip().split("\t")  # get field names myself (the numpy.genfromtxt gets rid of the "(" and ")" and also does not transfrm to lowercase)
            names = [self._string_simplify1(n) for n in names]
            f_data = str.encode(f.read())
            data = np.genfromtxt(io.BytesIO(f_data), names=names)
            data.dtype.names = names  # the genfromtxt gets rid of the "(" and ")"
            
            if files_read>0:
                if self.channels != data.dtype.names or self.points != data.shape[0]:
                    print("Warning: change in data structure in file %s. Stopping to read data." % fname)
                    break
            else:
                self.headers['channels'] = self.channels = data.dtype.names
                self.headers['points'] = self.points = data.shape[0]
            
            self.data_points.append({'data_headers': data_headers, 'data': data})

            files_read += 1
            f.close()

            
        if not read_somedata:
            print('Error: no data read.')
            return False
        
        # important info for the headers
        self.num_data_points = self.headers['num_data_points'] = len(self.data_points)
        
        # generate 3D numpy array for faster access and slicing
        self.data = self._generate_3D_data(self.data_points)
        
        print("%s files read." % (files_read))
            
            
    def _generate_3D_data(self,data_points):
        """once the data has been read in from the files, this function generates a 3D numpy array holding all the data"""
        
        data3D = np.zeros((len(data_points), len(self.channels), self.points))
        for i_data_point in range(len(data_points)):
            for i_channel in range(len(self.channels)):
                    data3D[i_data_point,i_channel,:] = data_points[i_data_point]['data'][self.channels[i_channel]]
        
        return data3D
        

    def fit_KPFM(self, x_limit=[]):
        """fits the KPFM parabolas and adds 'fit_type' and 'fit_coeffs' to the data_headers.
        Using the optional parameter x_limit a range can be defined where the fit is done (in x-axis units).
        Also extracts 'V*', 'df*', as well as the 'fit_sse' (sum of squares due to error), and 'fit_r2' (the r squared value). If x_limit is specified, also fit_sse_fullrange and fit_r2_fullrange will be calculated to represent the values for the full x range.
        These will be added to the data_header for each point.
        Also adds the amplitude mean and standard deviation ('amplitude_mean_(m)' and 'amplitude_stddev_(m)') - if the data exists. Further the fitted line and the residuals are added to the data sweeps."""
        
        # some sanity checks
        p0names = self.data_points[0]['data'].dtype.names
        if (not 'bias_[avg]_(v)' in p0names and not 'bias_(v)' in p0names):
            print("Error: Bias channel not found in the data. Cannot fit KPFM")
            return False
        if (not 'frequency_shift_[avg]_(hz)' in p0names and not 'frequency_shift_(hz)' in p0names):
            print("Error: Frequency shift channel not found in the data. Cannot fit KPFM.")
            return False
        
        for i,p in enumerate(self.data_points):
            if 'bias_[avg]_(v)' in p['data'].dtype.names:
                x = p['data']['bias_[avg]_(v)']
                if not 'bias_(v)' in p['data'].dtype.names:
                    self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'bias_(v)',p['data']['bias_[avg]_(v)'])  # just so that it will be easier to plot from the outside
            else:
                x = p['data']['bias_(v)']
            if 'frequency_shift_[avg]_(hz)' in p['data'].dtype.names:
                y = p['data']['frequency_shift_[avg]_(hz)']
                if not 'frequency_shift_(hz)' in p['data'].dtype.names:
                    self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'frequency_shift_(hz)',p['data']['frequency_shift_[avg]_(hz)'])  # just so that it will be easier to plot from the outside
            else:
                y = p['data']['frequency_shift_(hz)']
            
            if 'amplitude_[avg]_(m)' in p['data'].dtype.names:
                self.data_points[i]['data_headers']['amplitude_mean_(m)'] = np.mean(p['data']['amplitude_[avg]_(m)'])
                self.data_points[i]['data_headers']['amplitude_stddev_(m)'] = np.std(p['data']['amplitude_[avg]_(m)'])
            elif 'amplitude_(m)' in p['data'].dtype.names:
                self.data_points[i]['data_headers']['amplitude_mean_(m)'] = np.mean(p['data']['amplitude_(m)'])
                self.data_points[i]['data_headers']['amplitude_stddev_(m)'] = np.std(p['data']['amplitude_(m)'])
                
            if len(x_limit)==2:  # set limit for fits
                x_limit_i = [ (np.abs(x-x_limit[0])).argmin() , (np.abs(x-x_limit[1])).argmin()]
                if x_limit_i[0] > x_limit_i[1]:
                    x_limit_i[0], x_limit_i[1] = x_limit_i[1], x_limit_i[0]
                self.data_points[i]['data_headers']['fit_x_limit_i_start'] = x[x_limit_i[0]]
                self.data_points[i]['data_headers']['fit_x_limit_i_end'] = x[x_limit_i[1]]
                x_fit = x[x_limit_i[0]:x_limit_i[1]+1]
                y_fit = y[x_limit_i[0]:x_limit_i[1]+1]
            else:
                x_fit = x
                y_fit = y
            
            coeffs = np.polyfit(x_fit,y_fit,2)
            
            f = np.poly1d(coeffs)
            yhat = f(x)
            ybar = np.sum(y)/len(y)
            ssreg = np.sum((yhat-ybar)**2)
            sstot = np.sum((y - ybar)**2)
            sserr = np.sum((y - yhat)**2)
            yhat_fit = f(x_fit)
            ybar_fit = np.sum(y_fit)/len(y_fit)
            ssreg_fit = np.sum((yhat_fit-ybar_fit)**2)
            sstot_fit = np.sum((y_fit - ybar_fit)**2)
            sserr_fit = np.sum((y_fit - yhat_fit)**2)
            
            if 'fit_frequency_shift_(hz)' in self.data_points[i]['data'].dtype.names:
                self.data_points[i]['data']['fit_frequency_shift_(hz)'] = yhat
                self.data_points[i]['data']['fit_res_frequency_shift_(hz)'] = y - yhat  # residuals
            else:
                self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit_frequency_shift_(hz)', yhat)
                self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit_res_frequency_shift_(hz)', y - yhat)  # residuals
                
            self.data_points[i]['data_headers']['fit_type'] = "KPFM"
            self.data_points[i]['data_headers']['fit_coeffs'] = coeffs

            self.data_points[i]['data_headers']['fit_r2'] = ssreg_fit / sstot_fit
            self.data_points[i]['data_headers']['fit_sse'] = sserr_fit
            self.data_points[i]['data_headers']['fit_r2_fullrange'] = ssreg / sstot   # _fullrange takes the full range for error calculation (i.e. not taking into account x_limit)
            self.data_points[i]['data_headers']['fit_sse_fullrange'] = sserr
            
            self.data_points[i]['data_headers']['v*_(v)'] = -coeffs[1]/(2*coeffs[0])
            self.data_points[i]['data_headers']['df*_(hz)'] = coeffs[2] - coeffs[1]**2/(4*coeffs[0])
            
    
    def fit_IZ(self, oscillation_correction=False, x_limit=[]):
        """fits the IZ parabolas and adds 'fit_type' and 'fit_coeffs' to the data_headers.
        Also extracts the work function 'phi', as well as the 'fit_sse' (sum of squares due to error), and 'fit_r2' (the r squared value).
        Using the optional parameter x_limit a range can be defined where the fit is done (in x-axis units). If x_limit is specified, also fit_sse_fullrange and fit_r2_fullrange will be calculated to represent the values for the full x range.
        These will be added to the data_header for each point.
        Also adds the amplitude mean and standard deviation ('amplitude_mean_(m)' and 'amplitude_stddev_(m)') - if the data exists.
        Further the fitted line and the residuals are added to the data sweeps.
        
        If the parameter 'oscillation_correction' is set to True, another fit will be computed taking into account the z-oscillation (sensor amplitude).
        The fit then is a Bessel function of the first kind: I = I0 * exp(-2kz) * J0(2kA)
        """
        
        # some sanity checks
        p0names = self.data_points[0]['data'].dtype.names
        if (not 'z_[avg]_(m)' in p0names and not 'z_(m)' in p0names):
            print("Error: Z channel not found in the data. Cannot fit IZ.")
            return False
        if (not 'current_[avg]_(A)' in p0names and not 'current_(a)' in p0names):
            print("Error: Current channel not found in the data. Cannot fit IZ.")
            return False
        if oscillation_correction and (not 'amplitude_[avg]_(A)' in p0names and not 'amplitude_(a)' in p0names):
            print("Error: Amplitude channel not found in the data. Cannot fit IZ with oscillation correction.")
            return False
            
            
        for i,p in enumerate(self.data_points, x_limit=[]):
            if 'z_[avg]_(m)' in p['data'].dtype.names:
                x = p['data']['z_[avg]_(m)']
                if not 'z_(m)' in p['data'].dtype.names:
                    self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'z_(m)',p['data']['z_[avg]_(m)'])  # just so that it will be easier to plot from the outside
            else:
                x = p['data']['z_(m)']
            if 'current_[avg]_(a)' in p['data'].dtype.names:
                y_current = p['data']['current_[avg]_(a)']
                if not 'current_(a)' in p['data'].dtype.names:
                    self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'current_(a)',p['data']['current_[avg]_(a)'])  # just so that it will be easier to plot from the outside
            else:
                y_current = p['data']['current_(a)']

            if 'amplitude_[avg]_(m)' in p['data'].dtype.names:
                self.data_points[i]['data_headers']['amplitude_mean_(m)'] = np.mean(p['data']['amplitude_[avg]_(m)'])
                self.data_points[i]['data_headers']['amplitude_stddev_(m)'] = np.std(p['data']['amplitude_[avg]_(m)'])
                amplitude = p['data']['amplitude_[avg]_(m)']
            elif 'amplitude_(m)' in p['data'].dtype.names:
                self.data_points[i]['data_headers']['amplitude_mean_(m)'] = np.mean(p['data']['amplitude_(m)'])
                self.data_points[i]['data_headers']['amplitude_stddev_(m)'] = np.std(p['data']['amplitude_(m)'])
                amplitude = p['data']['amplitude_(m)']

            y = np.log(np.abs(y_current))

            if len(x_limit)==2:  # set limit for fits
                x_limit_i = [ (np.abs(x-x_limit[0])).argmin() , (np.abs(x-x_limit[1])).argmin()]
                if x_limit_i[0] > x_limit_i[1]:
                    x_limit_i[0], x_limit_i[1] = x_limit_i[1], x_limit_i[0]
                self.data_points[i]['data_headers']['fit_x_limit_i_start'] = x[x_limit_i[0]]
                self.data_points[i]['data_headers']['fit_x_limit_i_end'] = x[x_limit_i[1]]
                x_fit = x[x_limit_i[0]:x_limit_i[1]+1]
                y_fit = y[x_limit_i[0]:x_limit_i[1]+1]
            else:
                x_fit = x
                y_fit = y

            coeffs = np.polyfit(x_fit,y_fit,1)
            
            f = np.poly1d(coeffs)
            yhat = f(x)
            ybar = np.sum(y)/len(y)
            ssreg = np.sum((yhat-ybar)**2)
            sstot = np.sum((y - ybar)**2)
            sserr = np.sum((y - yhat)**2)
            yhat_fit = f(x_fit)
            ybar_fit = np.sum(y_fit)/len(y_fit)
            ssreg_fit = np.sum((yhat_fit-ybar_fit)**2)
            sstot_fit = np.sum((y_fit - ybar_fit)**2)
            sserr_fit = np.sum((y_fit - yhat_fit)**2)
            yhat_exp = np.exp(yhat)
            
            y_current_bar = np.sum(y_current)/len(y_current)
            if y_current_bar < 0: yhat_exp = -yhat_exp  # account for the sign of the current
            
            if 'fit_current_(a)' in self.data_points[i]['data'].dtype.names:
                self.data_points[i]['data']['log_current_(a)'] = y
                self.data_points[i]['data']['fit_log_current_(a)'] = yhat
                self.data_points[i]['data']['fit_res_log_current_(a)'] = y - yhat  # residuals
                self.data_points[i]['data']['fit_current_(a)'] = yhat_exp
                self.data_points[i]['data']['fit_res_current_(a)'] = y_current - yhat_exp  # residuals
            else:
                self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'log_current_(a)', y)
                self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit_log_current_(a)', yhat)
                self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit_res_log_current_(a)', y - yhat)  # residuals
                self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit_current_(a)', yhat_exp)
                self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit_res_current_(a)', y_current - yhat_exp)  # residuals
                
            self.data_points[i]['data_headers']['fit_type'] = "IZ"
            self.data_points[i]['data_headers']['fit_coeffs'] = coeffs
            
            self.data_points[i]['data_headers']['fit_r2'] = ssreg_fit / sstot_fit
            self.data_points[i]['data_headers']['fit_sse'] = sserr_fit
            self.data_points[i]['data_headers']['fit_r2_fullrange'] = ssreg / sstot  # _fullrange takes the full range for error calculation (i.e. not taking into account x_limit)
            self.data_points[i]['data_headers']['fit_sse_fullrange'] = sserr
            
            self.data_points[i]['data_headers']['phi_(j)'] = coeffs[0]**2 * (PLANCK_CONSTANT/(2*np.pi))**2 / (8 * MASS_ELECTRON)
            self.data_points[i]['data_headers']['phi_(ev)'] = self.data_points[i]['data_headers']['phi_(j)'] * CONV_J_EV

            if oscillation_correction:
                def log_curent_bessel(xx, k, logI0):
                    z = xx[0][:,0]
                    A = xx[0][:,1]
                    return logI0 + np.log(scipy.special.jv(0, -k*A)) + k*z    # jv(0, x) is the Bessel function of the first kind of order 0; -2*kappa = k
                
                coeffs2, pcov = scipy.optimize.curve_fit(f=log_curent_bessel, xdata=np.dstack([x,amplitude]), ydata=y, p0=[coeffs[0], coeffs[1]], maxfev=1000)  # p0 are initial values which are taken from the previous fit
                
                yhat2 = log_curent_bessel(np.dstack([x,amplitude]), coeffs2[0], coeffs2[1])
                ssreg2 = np.sum((yhat2-ybar)**2)
                sserr2 = np.sum((y - yhat2)**2)
                yhat_exp2 = np.exp(yhat2)
                
                if y_current_bar < 0: yhat_exp2 = -yhat_exp2  # account for the sign of the current
                
                if 'fit2_current_(a)' in self.data_points[i]['data'].dtype.names:
                    self.data_points[i]['data']['fit2_log_current_(a)'] = yhat2
                    self.data_points[i]['data']['fit2_res_log_current_(a)'] = y - yhat2  # residuals
                    self.data_points[i]['data']['fit2_current_(a)'] = yhat_exp2
                    self.data_points[i]['data']['fit2_res_current_(a)'] = y_current - yhat_exp2  # residuals
                else:
                    self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit2_log_current_(a)', yhat2)
                    self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit2_res_log_current_(a)', y - yhat2)  # residuals
                    self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit2_current_(a)', yhat_exp2)
                    self.data_points[i]['data'] = recfunctions.append_fields(self.data_points[i]['data'],'fit2_res_current_(a)', y_current - yhat_exp2)  # residuals
                    
                self.data_points[i]['data_headers']['fit2_type'] = "IZosc"
                self.data_points[i]['data_headers']['fit2_coeffs'] = coeffs2
                
                self.data_points[i]['data_headers']['fit2_r2'] = ssreg2 / sstot
                self.data_points[i]['data_headers']['fit2_sse'] = sserr2
                
                self.data_points[i]['data_headers']['phi2_(j)'] = coeffs2[0]**2 * (PLANCK_CONSTANT/(2*np.pi))**2 / (8 * MASS_ELECTRON)
                self.data_points[i]['data_headers']['phi2_(ev)'] = self.data_points[i]['data_headers']['phi2_(j)'] * CONV_J_EV

                
    def fit_IZosc(self):
        """shorthand for fit_IZ(oscillation_correction=True)"""
        if 'amplitude_[avg]_(m)' in self.data_points[0]['data'].dtype.names or 'amplitude_(m)' in self.data_points[0]['data'].dtype.names:
            self.fit_IZ(oscillation_correction=True)
        else:
            print('Error: point-per point amplitude information not found, but is necessary for the IZ fit with oscillation correction.')

                
    def _string_simplify1(self,str):
        """simplifies a string (i.e. removes replaces space for "_", and makes it lowercase"""
        return str.replace(' ','_').lower()

    def _string_simplify2(self,str):
        """simplifies a string (i.e. removes replaces space for "_", and makes it lowercase"""
        return str.replace(' ','_').lower()
    


class PlotData(object):
    """Class to read grid date from binary or ascii files. Also fits the KPFM parabola and the exponential IZ curves. """
    
    def __init__(self, griddata):
        """we provide a griddate object as input"""
        # populate from an GridData object
        griddata_ = copy.deepcopy(griddata)  # we want copies, so that we do not alter the original griddata object
        self.filename = griddata_.filename
        self.headers = griddata_.headers
        self.data_points = griddata_.data_points
        
        self.num_data_points = griddata_.num_data_points
        self.channels = griddata_.channels
        self.points = griddata_.points
        self.data = griddata_.data
        
        self.im_extent = ()      # x and y span of image (x_min, x_max, y_min, y_max)
        self.im_delta = (1,1)    # spacing between points
        self.grid_dim = ()       # grid dimensions in pixels
        self.grid_center = ()    # center coordinates for grid
        self.grid_extent = ()    # grid span in x and y direction
        self.grid_rotation = -1  # grid rotation
        
        self.axis_length = False  # true if we have physical length info for the data
        
        self.test = []  # for testing events
        
        self.fig = None
        self.gs = None            # gridspec
        self.plots = []           # grid plots
        self.plots2D = []         # 2D graphs
        self.plots2D2 = []        # 2D graphs for 2nd graph
        self.plot_markers = []    # markers on grid plots
        self.selected_index = 0   # selected index to plot 2D plots
        self.plot_legends = []    # legends for 2D graph

        self.plot_sweep_signal = None          # channel that is swept
        self.plot_xaxis = None                 # xaxis channel
        self.plot_sweep_channels_selected = []
        self.plot_sweep_channel_res = None      # channel for residuals
        

    def _length_to_indices(self,coords_index):
        """converts length coordinates to index coordinates"""
        return (int(coords_index[0]/self.im_delta[0]), int(coords_index[1]/self.im_delta[1]))
    
    def _index_to_length(self,coords_length):
        """converts index coordinates to length coordinates"""
        return (coords_length[0]*self.im_delta[0], coords_length[1]*self.im_delta[1])
        
    def _indices_to_index(self,coords_index):
        """converts pixel indices to data point index"""
        return coords_index[1] * self.grid_dim[0] + coords_index[0]
        
    def _index_to_indices(self,index):
        """converts data point index to pixel indices"""
        return (index % self.grid_dim[0], int(np.floor(index/self.grid_dim[0])))
        
    def _indices_check_bounds(self, coords_index):
        """checks and adjust bounds of pixel coordinates"""
        x = min(coords_index[0],self.grid_dim[0])
        x = max(coords_index[0],0)
        y = min(coords_index[1],self.grid_dim[1])
        y = max(coords_index[1],0)
        return (x,y)

    def _point_distance(self, p1, p2, absolute=True):
        """calculates distance between two points. If absolute is True, then the absolute value will be returned."""
        d = np.sqrt((p2['data_headers']['x_(m)']-p1['data_headers']['x_(m)'])**2 + (p2['data_headers']['y_(m)']-p1['data_headers']['y_(m)'])**2)
        if absolute:
            return abs(d)
        else:
            return d


    def _calc_grid_info(self):
        """Computes grid information"""
        warn = []
        self.axis_length = False
        if 'grid_settings' in self.headers:
            self.grid_center = (self.headers['grid_settings'][0],self.headers['grid_settings'][1])
            self.grid_extent = (self.headers['grid_settings'][2],self.headers['grid_settings'][3])
            self.grid_rotation = self.headers['grid_settings'][4]
            self.grid_dim = self.headers['grid_dim']
            self.im_extent = (0,self.grid_extent[0]*1e9,0,self.grid_extent[1]*1e9)  # in nanometers
            self.im_delta = ((self.im_extent[1]-self.im_extent[0])/self.grid_dim[0], (self.im_extent[3]-self.im_extent[2])/self.grid_dim[1])
            self.axis_length = True
        else:
            self.im_delta = (1,1)  # for points. we will try to extract the length later
            
            # analyze length data
            warn.append("grid_unknown")
            dx = self._point_distance(self.data_points[0],self.data_points[1])
            # initial values, should be changed in the loop below
            n_sqrt = int(np.sqrt(len(self.data_points)))
            nx, ny = n_sqrt, n_sqrt
            dy = self._point_distance(self.data_points[ny],self.data_points[0])
            for i in range(len(self.data_points[:-1])):
                d = self._point_distance(self.data_points[i],self.data_points[i+1])
                if abs(d-dx) > 1e-12: # looks like there is a dy step now
                    nx = i+1
                    dy = self._point_distance(self.data_points[nx],self.data_points[0])
                    ny = int(np.ceil(len(self.data_points)/nx))
                    break

            self.grid_dim = (nx, ny)
            self.im_extent = (0, nx*dx*1e9, 0, ny*dy*1e9)  # estimating the extent of the image, convert to nanometers
            self.im_delta = (dx*1e9, dy*1e9)  # converting to nm
            self.grid_center = (self.data_points[0]['data_headers']['x_(m)'] + nx/2 * dx, self.data_points[0]['data_headers']['y_(m)'] + ny/2 * dy)
            dx_p1p0 = self.data_points[1]['data_headers']['x_(m)']-self.data_points[0]['data_headers']['x_(m)']  # x-difference between point 1 and point 0
            dy_p1p0 = self.data_points[1]['data_headers']['y_(m)']-self.data_points[0]['data_headers']['y_(m)']  # y-difference between point 1 and point 0
            self.grid_rotation = (np.arctan2(dy_p1p0, dx_p1p0) * 180 / np.pi + 360) % 360 # the "+360%360" ensures that the angle is always positive
            self.axis_length = True

            if (nx * ny != len(self.data_points)):  # we can try to extract some length data
                warn.append("grid_unknown_problem")

        return warn
        

    def _widgets_update_vars(self):
        """updates class variables with GUI settings"""
        self.selected_index = self.slider_point.value - 1
        self.plot_sweep_channels_selected = self.select_channels.value
        self.plot_xaxis = self.select_xaxis.value
        self.plot_sweep_channels_selected2 = self.select_channels2.value
        self.plot_xaxis2 = self.select_xaxis2.value
    
    def _widgets_grid(self,name=None,state=False):
        """toggle grid in plots"""
        if not name:  # no name and no state given, so we switch
            self.button_grid.value = not self.button_grid.value
            state = self.button_grid.value
        for p in self.plots:
            p.grid(state)
        if PYTHON_VERSION==2: plt.draw()
            
    def _widgets_legend(self,name=None,state=False):
        """toggle legends in plots"""
        if not name:  # no name and no state given, so we switch
            self.button_legend.value = not self.button_legend.value
            state = self.button_legend.value
        for legend in self.plot_legends:
            legend.set_visible(state)
        if PYTHON_VERSION==2: plt.draw()

    def _slider_change(self):
        """when slider changes, update markers and plot"""
        self.selected_index = self.slider_point.value - 1
        self._widgets_change()
        
    def _widgets_change_marker(self):
        """when marker display changes"""
        self._widgets_marker()
    
    def _widgets_change(self):
        """when a widget changes, update markers and plot"""
        self.slider_point.value = self.selected_index + 1
        self._widgets_update_vars()
        self._widgets_marker(redraw=False)
        self._widgets_plot_change()
        
    def _widgets_plot_change(self):
        """widgets regarding plot changes, update plot"""
        self._widgets_update_vars()
        self.plot_sweep_data()
        if PYTHON_VERSION==2: plt.draw()
        
    def _widgets_marker(self, redraw=True):
        """toggle marker in plots"""
        x,y = self._index_to_length(self._index_to_indices(self.slider_point.value-1))
        w,h = self._index_to_length((1,1))
        for i in reversed(range(len(self.plot_markers))):
            self.plot_markers[i].remove()
            del self.plot_markers[i]
        if self.button_marker.value:
            for p in self.plots:
                self.plot_markers.append(p.add_patch(matplotlib.patches.Rectangle((x, y), w, h, fill=False, snap=False, edgecolor='#900000', linewidth=1)))
        if self.button_marker_circle.value:
            for p in self.plots:
                w2 = w*2
                self.plot_markers.append(p.add_patch(matplotlib.patches.Circle((x+w/2,y+h/2), w2, fill=True, snap=False, facecolor='#f00000',linewidth=1, edgecolor='#900000', alpha=0.33)))
        if PYTHON_VERSION==2 and redraw: plt.draw()
            
    def _plot_onclick(self,event):
        """onclick event for plot"""
        if event.xdata == None or event.ydata == None: return False
        if self.axis_length:
            pixel_coord = self._length_to_indices((event.xdata, event.ydata))  # convert to pixel index
        else:
            pixel_coord = (int(event.xdata), int(event.ydata))  # x and y coordinates are point indices
            
        pixel_coord = self._indices_check_bounds(pixel_coord)  # check if we are out of range, and adjust if necessary
        self.selected_index = self._indices_to_index(pixel_coord)
        
        self._widgets_change()
        
    def _plot_onkeypress(self,event):
        """changes selected points according to the cursor arrows"""
        if event.key == "right":
            new_index = self.selected_index + 1
        elif event.key == "left":
            new_index = self.selected_index - 1
        elif event.key == "up":
            new_index = self.selected_index + self.grid_dim[0]
        elif event.key == "down":
            new_index = self.selected_index - self.grid_dim[0]
        elif event.key == "m":
            self.button_marker.value = not self.button_marker.value
            self._widgets_marker()
            return
        elif event.key == "M":
            self.button_marker_circle.value = not self.button_marker_circle.value
            self._widgets_marker()
            return
        elif event.key == "i":
            self._widgets_legend()
            return
        elif event.key == "g":
            self._widgets_grid()
            return
        else:
            return
        
        if new_index >=0 and new_index < len(self.data_points): self.selected_index = new_index
        self._widgets_change()
        
        
    def plot_options(self):
        """displays options for the plot as ipython widgets"""
        
        self.button_grid = ipywidgets.widgets.ToggleButton(description='Grid',value=False, width=120)
        self.button_legend = ipywidgets.widgets.ToggleButton(description='Legend',value=True, width=120)
        self.button_marker = ipywidgets.widgets.ToggleButton(description='Marker',value=False, width=120)
        self.button_marker_circle = ipywidgets.widgets.ToggleButton(description='Marker+',value=False, width=120)
        self.button_plot2 = ipywidgets.widgets.ToggleButton(description='2nd graph',value=True, width=120)
        self.slider_point = ipywidgets.widgets.IntSlider(description='Point:',value=self.selected_index+1,min=1,max=len(self.data_points), width=200)
        
        self.plot_sweep_channels = self.data_points[0]['data'].dtype.names
        self.plot_sweep_signal = self.plot_sweep_channels[0]
        self.plot_sweep_channels2 = self.data_points[0]['data'].dtype.names
        if 'sweep_signal' in self.headers:
            self.plot_sweep_signal = self.headers['sweep_signal']
        elif 'fit_type' in self.data_points[0]['data_headers']:
            if self.data_points[0]['data_headers']['fit_type'] == 'KPFM':
                self.plot_sweep_signal = 'bias_(v)'
            elif self.data_points[0]['data_headers']['fit_type'] == 'IZ':
                self.plot_sweep_signal = 'z_(m)'
        self.plot_sweep_channels_selected = [self.plot_sweep_channels[0]]
        if 'fit_type' in self.data_points[0]['data_headers']:
            if self.data_points[0]['data_headers']['fit_type'] == 'KPFM':
                self.plot_sweep_channels_selected = ["frequency_shift_(hz)","fit_frequency_shift_(hz)"]
                self.plot_sweep_channels_selected2 = ["fit_res_frequency_shift_(hz)"]
            elif self.data_points[0]['data_headers']['fit_type'] == 'IZ':
                self.plot_sweep_channels_selected = ["log_current_(a)","fit_log_current_(a)"]
                self.plot_sweep_channels_selected2 = ["fit_res_log_current_(a)"]
                if 'fit2_type' in self.data_points[0]['data_headers']:
                    if self.data_points[0]['data_headers']['fit2_type'] == 'IZosc':
                        self.plot_sweep_channels_selected = ["log_current_(a)","fit2_log_current_(a)"]
                        self.plot_sweep_channels_selected2 = ["fit2_res_log_current_(a)"]
        self.plot_xaxis = self.plot_sweep_signal
        self.plot_sweep_signal2 = self.plot_sweep_signal
        self.plot_xaxis2 = self.plot_sweep_signal2

        self.select_xaxis = ipywidgets.widgets.Dropdown(description="x axis:",options=self.plot_sweep_channels,value=self.plot_sweep_signal, width=200)
        self.select_channels = ipywidgets.widgets.SelectMultiple(description="Channels:",options=self.plot_sweep_channels,value=self.plot_sweep_channels_selected, width=200)
        self.select_xaxis2 = ipywidgets.widgets.Dropdown(description="x axis:",options=self.plot_sweep_channels,value=self.plot_sweep_signal2, width=200)
        self.select_channels2 = ipywidgets.widgets.SelectMultiple(description="Channels:",options=self.plot_sweep_channels,value=self.plot_sweep_channels_selected2, width=200)
        
        tab_options = ipywidgets.widgets.HBox([
            ipywidgets.widgets.VBox([self.slider_point,ipywidgets.widgets.HBox([self.button_marker,self.button_marker_circle]),ipywidgets.widgets.HBox([self.button_grid,self.button_legend]),self.button_plot2]),
            ipywidgets.widgets.VBox([self.select_xaxis,self.select_channels]),
            ipywidgets.widgets.VBox([self.select_xaxis2,self.select_channels2])
        ], padding=12)
        
        # events
        self.button_grid.on_trait_change(self._widgets_grid, 'value')
        self.button_legend.on_trait_change(self._widgets_legend, 'value')
        self.slider_point.on_trait_change(self._slider_change)  # slider needs a special one, because selected_index is updated
        self.button_marker.on_trait_change(self._widgets_change_marker)
        self.button_marker_circle.on_trait_change(self._widgets_change_marker)
        self.select_xaxis.on_trait_change(self._widgets_plot_change)
        self.select_channels.on_trait_change(self._widgets_plot_change)
        self.select_xaxis2.on_trait_change(self._widgets_plot_change)
        self.select_channels2.on_trait_change(self._widgets_plot_change)
        self.button_plot2.on_trait_change(self._widgets_plot_change)
        
        self.plot_sweep_data()
        
        return tab_options


    def plot_sweep_data(self, index=None, xaxis=None, channels=None, xaxis2=None, channels2=None, graph2=None):
        """plots sweep data in one or two 2D graphs"""
        if not index:
            index = self.selected_index
        if not xaxis:
            xaxis = self.plot_xaxis
        if not channels:
            channels = self.plot_sweep_channels_selected
        if not xaxis2:
            xaxis2 = self.plot_xaxis2
        if not channels2:
            channels2 = self.plot_sweep_channels_selected2
        if not graph2:
            graph2 = self.button_plot2.value


        # delete old data
        for plots2D_ in (self.plots2D, self.plots2D2):
            for p in plots2D_:
                if p in self.fig.get_axes():  # we have to be careful here, as sometimes we reuse the same plot for the channels (i.e. for fit and the respetive original data)
                    for i in reversed(range(len(p.lines))):
                        p.lines[i].remove()
            for i,p in enumerate(plots2D_):
                if p in self.fig.get_axes(): self.fig.delaxes(plots2D_[i])

        self.plot_legends = []        
        self.plots2D = []
        self.plots2D2 = []
        if graph2:
            self.plots2D.append(self.fig.add_subplot(self.gs[-2,:]))
            if xaxis==xaxis2:
                self.plots2D2.append(self.fig.add_subplot(self.gs[-1,:], sharex=self.plots2D[0]))
            else:
                self.plots2D2.append(self.fig.add_subplot(self.gs[-1,:]))
        else:
            self.plots2D.append(self.fig.add_subplot(self.gs[-2:,:]))

        for xaxis_,channels_,plots2D_ in zip([xaxis,xaxis2],[channels,channels2],[self.plots2D,self.plots2D2]):
            if not graph2 and plots2D_==self.plots2D2: continue  # we do not need to do anything for the second plot if it should not be displayed
            x = self.data_points[index]['data'][xaxis_]
            lw = 2
            ms = 4
            ls = "-"
            m = "."
            lines, labels = [], []  # for the combined legend
            num_sharey = 0  # number of shared y-axes (for fit and original data)
            if xaxis_ != self.plot_sweep_signal: lw = 0  # do not show connecting lines if the xaxis is not the sweep signal
            for i,c in enumerate(channels_):
                y = self.data_points[index]['data'][c]
                color = sns.color_palette()[i % len(sns.color_palette())]
                found_similar = False
                if i>0:
                    p = None
                    for ii in range(i):  # but not if there is a pair of fitted and original values
                        if channels_[ii].replace('fit_','') == c.replace('fit_','') or channels_[ii].replace('fit2_','') == c.replace('fit2_',''):
                            #p = plots2D[ii]_.twinx().twiny()
                            p = plots2D_[ii]  # i am adding the same plot here
                            num_sharey += 1
                            found_similar = True
                    if not p:  p = plots2D_[0].twinx() # plot with different y axis, otherwise we have scaling issues
                    plots2D_.append(p)
                plots2D_[i].plot(x, y, label=c, color=color, ls=ls, marker=m, lw=lw, ms=ms)
                plots2D_[i].grid(False)
                if 'fit_x_limit_i_start' in self.data_points[index]['data_headers'].keys():
                    plots2D_[i].axvline(self.data_points[index]['data_headers']['fit_x_limit_i_start'], ls='dashed', color="#aaaaaa")
                    plots2D_[i].axvline(self.data_points[index]['data_headers']['fit_x_limit_i_end'], ls='dashed', color="#aaaaaa")
                
                # make x-axis label if (i) there is no second graph, or (ii) the axis labels are different, or (iii) we are plotting the second graph
                if not graph2 or xaxis != xaxis2 or plots2D_== self.plots2D2:
                    plots2D_[i].set_xlabel(xaxis_)
                if not found_similar:
                    if i<2+num_sharey: plots2D_[i].set_ylabel(c)  # only show the first two labels
                    
            # combined legend
            for p in self.fig.get_axes():  # this way I count the plots only once
                if p in plots2D_:
                    lines += p.get_legend_handles_labels()[0]
                    labels += p.get_legend_handles_labels()[1]
            self.plot_legends.append(plots2D_[0].legend(lines, labels, loc=0))
        
        
    def plot_channels(self,channels, num_rows=3, cmap='Blues_r'):  # cmap = "RdBu"
        """Plots a grid of several channels. The parameter num_rows specifies how many rows per line,
        cmap is optional and specifies the colormap to use (can also be a list)"""

        # clean up channels that are not in the data
        channels_notfound = []
        for i,c in enumerate(channels):
            if not c in self.data_points[i]['data_headers']:
                channels_notfound.append(i)
        channels_notfound_names = [channels[i] for i in channels_notfound]
        channels = [c for i,c in enumerate(channels) if i not in channels_notfound]
        if len(channels)<1:
            print("Error: no channels to plot.")

        num_channels = len(channels)
        num_cols = int(np.ceil(num_channels/num_rows))

        self.fig = plt.figure(figsize=(2.8*num_rows,2.5*(num_cols+2*0.8)))
        plots = []
        plots_widgets = []

        warn = self._calc_grid_info()
        
        # if there is incomplete data, fill up the missing points with the last point
        points_appended = 0
        while self.grid_dim[0] * self.grid_dim[1] > len(self.data_points):
            self.data_points.append(self.data_points[-1])
            points_appended += 1
            
        # plot channels
        #gs = matplotlib.gridspec.GridSpec(num_cols,num_rows+1, width_ratios=[1]*num_rows+[0.5], hspace=0.5, wspace=0.5)
        gs = matplotlib.gridspec.GridSpec(num_cols+2,num_rows, width_ratios=[1]*num_rows, height_ratios=[1]*num_cols+[0.8]+[0.8], hspace=0.5, wspace=0.5)
        for i,channel in enumerate(channels):
            z = [self.data_points[ii]['data_headers'][channel] for ii in range(len(self.data_points))]
            z = np.array(z)
            z = z.reshape(self.grid_dim[1],self.grid_dim[0])
            
            if i==0:
                plots.append(self.fig.add_subplot(gs[0,0]))
            else:
                plots.append(self.fig.add_subplot(gs[int(np.floor(i/num_rows)),i % num_rows], sharex=plots[0], sharey=plots[0]))
            if isinstance(cmap, (list, tuple)):
                cmap_ = cmap[i % len(cmap)]
            else:
                cmap_=cmap
            img = plots[i].imshow(z, cmap=cmap_, interpolation='nearest', origin='lower',picker=True)
            if self.axis_length:
                plt.setp(img, extent=self.im_extent)
                plots[i].set_xlabel("nm")
                #plots[i].set_ylabel("nm")
            else:
                plots[i].set_xlabel("point")
                #plots[i].set_ylabel("point")
            
            plots[i].grid(False)
            plots[i].set_title(channel)
            cbar = plt.colorbar(img,fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8) 
        
        # some figure styling
        self.fig.suptitle(self.filename, y=0.98)
        #self.fig.tight_layout(pad=pad)
        
        if "grid_unknown" in warn: print("Warning: Do not know the dimensions of the image. Assuming a %s x %s grid of %0.2f x %0.2f nm." % (self.grid_dim[0], self.grid_dim[1], self.im_extent[1], self.im_extent[3]))
        if "grid_unknown_problem" in warn: print("Warning: The assumed grid does not seem right (maybe the data is incomplete). The data representation might not be correct.")
        if points_appended>0 :  print("Warning: The data seems incomplete. Appended %s points to make it a rectangular grid." % points_appended)
        if len(channels_notfound)>0: print("Warning: The following channels have not been found in the data: %s." % ', '.join(channels_notfound_names))
        if self.axis_length:  print("Image center: %0.4f, %0.4f nm. Rotation: %0.2f degrees." % (self.grid_center[0] * 1e9, self.grid_center[1] * 1e9, self.grid_rotation))
        
        # events
        cid = self.fig.canvas.mpl_connect('button_press_event', self._plot_onclick)
        cid2 = self.fig.canvas.mpl_connect('key_press_event', self._plot_onkeypress)
        
        self.plots = plots
        self.gs = gs
        
        return self.fig
        
        
if __name__=="__main__":
    pass