"""
Needed to make a lot of tables in the terminal and there is no support for it, so I made this program.
There are now various useful general functions in here, and some functions made specifically for the QPE vs TDE paper
"""

import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
from tqdm import tqdm

def print_table(a, header=None, title=None, space_between_columns=2, space_between_rows=0, borders=1, header_color="yellow", border_color="grey", override_length=None):
    """
    Nicely print out a table

    a: array to print
    header: either an array of column names or a boolean. True -> first row will be header
    title: string, title that will be centered above the table
    space_between_columns: int, self-explanatory
    space_between_rows: int, self-explanatory
    """
    a = np.array(a, dtype=str)
    if type(header) == str or (type(header) == bool and header):
        pass
    elif (type(header) == bool and header == False) or header == None:
        header = None
    else:
        a = np.vstack((header, a))

    #Initialize the ascii characters to create the table depending on the borders parameter:
    if borders == None or borders == 0 or borders == False or borders == "none":
        characters = [" "," "," "," "," "," "]
    elif borders == "bold" or borders == 2:
        characters = ["═","║","╔","╗","╚","╝"]
    elif borders == 1 or borders == True or borders == "normal":
        characters = ["─","│","┌","┐","└","┘"]
    else:
        if type(borders) == str and len(borders) == 1:
            characters = [*(borders*6)]
        else:
            raise ValueError(f"Border style '{borders}' does not exist, use the keyword 'none', 'normal' or 'bold'.")
    
    possible_colors = ["black","red","green","yellow","blue","magenta","cyan","white"]
    #Initialize the colors:
    #Header color
    if header_color == None or header_color == "grey":
        header_color = "0"
    elif type(header_color) == str:
        header_color = header_color.lower()
        if header_color in possible_colors:
            header_color = str(possible_colors.index(header_color)+30)
        else:
            print(f"Color '{header_color}' not implemented, defaulting to grey.\nPossible colors are: {['grey']+possible_colors}")
            header_color = "0"
    else:
        raise ValueError(f"Parameter 'header_color' needs to be a string.")
    #Borders color
    if border_color == None or border_color == "grey":
        border_color = "0"
    elif type(border_color) == str:
        border_color = border_color.lower()
        if border_color in possible_colors:
            border_color = str(possible_colors.index(border_color)+30)
        else:
            print(f"Color '{border_color}' not implemented, defaulting to grey.\nPossible colors are: {['grey']+possible_colors}")
            border_color = "0"
    else:
        raise ValueError("Parameter 'border_color' needs to be a string.")

    for i in range(len(characters)):
        characters[i] = f"\x1b[{border_color}m{characters[i]}\x1b[0m"

    #Replace (None) elements with "-":
    a[a == "None"] = "-"

    #Get longest string in each column:
    column_maxes = []
    vfunc = np.vectorize(lambda x: len(x))
    a_lens = vfunc(a)
    for i in range(a.shape[1]):
        column_maxes.append(np.max(a_lens[:,i]))
    if override_length != None:
        column_maxes = override_length
    total_length = np.sum(column_maxes)+(len(column_maxes)-1)*space_between_columns #To include spaces between each column

    #Actually start printing table:
    top_and_bottom_bounds = (characters[2]+characters[0]*(total_length+2)+characters[3], characters[4]+characters[0]*(total_length+2)+characters[5])
    print()
    usable_length = total_length+4
    if title != None:
        title = floor((usable_length-len(title))/2)*" "+title+ceil((usable_length-len(title))/2)*" "
        print(f"\x1b[{header_color}m{title}\x1b[0m")
    print(top_and_bottom_bounds[0])
    #Print each row:
    for row in range(a.shape[0]):
        row_string = ""
        for column in range(a.shape[1]):
            row_string += a[row, column] + " "*(column_maxes[column]-a_lens[row,column])
            if column < a.shape[1]-1:
                row_string += " "*space_between_columns
        if row == 0 and header != None:
            row_string = f"\x1b[{header_color}m{row_string}\x1b[0m"
        row_string = f"{characters[1]} {row_string} {characters[1]}"
        if row != (a.shape[0]-1):
            row_string += f"\n{characters[1]} {' '*(total_length)} {characters[1]}"*space_between_rows
            if row == 0 and header != None:
                row_string += f"\n{characters[1]} {' '*(total_length)} {characters[1]}"
        print(row_string)
    print(top_and_bottom_bounds[1])
    print()








def myCornerPlot(data, labels=None, bw_adjust=1, units=None, fontsize=15, smoothness=6, linewidth=3, extremums=None, background_colors=["#f0f0f0","#969696","#252525"], levels=4, markersize=8, refCat=None, columns_compare=None, save_plot=None):
    """
    Make corner plots with kernel histograms
    data should be [data_set1, data_set2, ...] each containing multiple parameters
    """
    for i in range(len(data)-1):
        assert len(data[i]) == len(data[i+1])

    # labels are now required (just put empty ones if you don't want them to appear)
    if extremums is None:
        extremums = {}
    for label in labels:
        if label not in extremums:
            extremums[label] = None
    
    
    # Create the plot axes:
    fig = plt.figure(figsize=(10,8))
    plot_size = len(data[0])
    if refCat is not None:
        assert plot_size == len(columns_compare)
    if labels is not None:
        assert plot_size == len(labels)
    if units is not None:
        assert plot_size == len(units)
    hist_axes = []
    corner_axes = []
    for i in range(plot_size):
        hist_axes.append(plt.subplot(plot_size,plot_size,i*plot_size+i+1))
        corner_axes.append([])
        for k in range(plot_size-(i+1)):
            if i == 0:
                corner_axes[i].append(plt.subplot(plot_size,plot_size,(i+k+1)*plot_size+(i+1),sharex=hist_axes[i]))
            else:
                corner_axes[i].append(plt.subplot(plot_size,plot_size,(i+k+1)*plot_size+(i+1),sharex=hist_axes[i],sharey=corner_axes[i-1][k+1]))
            if k != plot_size-(i+1)-1:
                corner_axes[i][k].get_xaxis().set_visible(False)
            if i != 0:
                corner_axes[i][k].get_yaxis().set_visible(False)
            corner_axes[i][k].xaxis.set_tick_params(labelsize=fontsize-2)
            corner_axes[i][k].yaxis.set_tick_params(labelsize=fontsize-2)
        if i == plot_size-1:
            hist_axes[i].get_yaxis().set_visible(False)
            hist_axes[i].xaxis.set_tick_params(labelsize=fontsize-2)
        else:
            hist_axes[i].get_xaxis().set_visible(False)
            hist_axes[i].get_yaxis().set_visible(False)

    # Show data in each plot:

    
    #Plot kernel histograms:
    for i in range(plot_size):
        if labels is not None:
            hist_axes[i].set_title(labels[i], fontsize=fontsize)
        x_min, x_max = np.min(data[0][i]), np.max(data[0][i])
        for j in range(len(data)):
            x_min = np.min(data[j][i]) if x_min > np.min(data[j][i]) else x_min
            x_max = np.max(data[j][i]) if x_max < np.max(data[j][i]) else x_max
        if refCat is not None:
            x_min = x_min if x_min < np.min(refCat[f"col_{columns_compare[i]}"]) else np.min(refCat[f"col_{columns_compare[i]}"])
            x_max = x_max if x_max > np.max(refCat[f"col_{columns_compare[i]}"]) else np.max(refCat[f"col_{columns_compare[i]}"])
        if extremums[labels[i]] is not None:
            x_min, x_max = extremums[labels[i]]
        for j in range(len(data)-1):
            X_plot = np.linspace(x_min, x_max, 1000)[:,np.newaxis]
            bandwidth = np.abs(x_max-x_min)/smoothness
            if (refCat is not None) and j == 0:
                kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(np.array(refCat[f"col_{columns_compare[i]}"])[:,np.newaxis])
                log_dens = kde.score_samples(X_plot)
                #hist_axes[i].fill_between(X_plot[:, 0], np.exp(log_dens), fc="grey", alpha=0.4)
                hist_axes[i].plot(X_plot[:, 0], np.exp(log_dens), color="black", linewidth=linewidth)
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(data[j][i][:,np.newaxis])
            log_dens = kde.score_samples(X_plot)
            hist_axes[i].fill_between(X_plot[:, 0], np.exp(log_dens), fc=["blue","red","orange"][j%3], alpha=[0.4,0.4][j])


 
    for i in range(plot_size):
        for k in range(len(corner_axes[i])):
            for j in range(len(data)):
                corner_axes[i][k].plot(data[j][i], data[j][i+k+1], ["o","*","*"][j%3], color=["blue","red","red"][j%3], markersize=[markersize,markersize-1,markersize-1][j%3])
            if refCat is not None:
                sns.kdeplot(refCat, bw_adjust=bw_adjust, x=f"col_{columns_compare[i]}", y=f"col_{columns_compare[i+k+1]}", fill=True, levels=levels, colors=background_colors, ax=corner_axes[i][k])
                #sns.kdeplot(refCat, x=f"col_{columns_compare[i]}", y=f"col_{columns_compare[i+k+1]}", fill=False, levels=levels, linewidths=0.5, color="black", ax=corner_axes[i][k])
        print_color(f"{labels[i]} :")
        print(f"QPE: {(np.min(data[0][i]), np.median(data[0][i]), np.max(data[0][i]))}")
        print(f"TDE: {(np.min(data[1][i]), np.median(data[1][i]), np.max(data[1][i]))}")
        print(f"ref: {(np.min(refCat[f'col_{columns_compare[i]}']), np.median(refCat[f'col_{columns_compare[i]}']), np.max(refCat[f'col_{columns_compare[i]}']))}")
    
    # Make units labels and set axis limits:
    if units is None:
        units = [" " for i in range(plot_size)]
    for i in range(plot_size):
        if i < plot_size-1:
            if i > 0:
                corner_axes[0][i-1].set_ylabel(labels[i]+"\n"+units[i], fontsize=fontsize-1)
                if extremums[labels[i]] is not None:
                    corner_axes[0][i-1].set_ylim(*extremums[labels[i]])
            corner_axes[i][-1].set_xlabel(labels[i]+"\n"+units[i], fontsize=fontsize-1)
            if extremums[labels[i]] is not None:
                corner_axes[i][-1].set_xlim(*extremums[labels[i]])
        else:
            corner_axes[0][i-1].set_ylabel(labels[i]+"\n"+units[i], fontsize=fontsize-1)
            if extremums[labels[i]] is not None:
                corner_axes[0][i-1].set_ylim(*extremums[labels[i]])
            hist_axes[i].set_xlabel(labels[i]+"\n"+units[i], fontsize=fontsize-1)
            if extremums[labels[i]] is not None:
                hist_axes[i].set_xlim(*extremums[labels[i]])
    
    plt.subplots_adjust(left=0.095, bottom=0.1, right=0.99, top=0.955, wspace=0, hspace=0)
    if save_plot is not None:
        plt.savefig(f"{save_plot}.pdf")
    plt.show()
    return



if __name__ == "__main__":
    data = np.array([["potato", 5178, 13095, 3151],
            ["123", None, 1023, 51515],
            ["potato", 5178, 13012495, 51515],
            ["123", 123, 1024443, 51515],
            ["potaawddwto", 5178, 13095, 51515],
            ["123", "something", 1023, 51515],
            ["potato", 5178, 13095, 51515],
            ["123", None, 1023, 51515]])
    print_table(data,
                space_between_columns=4,
                space_between_rows=0,
                header=[f"Column{i}" for i in range(data.shape[1])],
                title="My Table",
                borders="bold",
                header_color="yellow",
                border_color="blue",
                )
    





def makeLatexTable(names, redshifts, r50s, n_sersics, bt_ratios, ssmds, M_stars, M_bhs, references=None, filename="latexTable.txt", verbose=False):
    """
    columns are: NAME, REDSHIFT, R_50, N_SERSIC, BT_RATIO, SSMD, M_stars

    makes a .txt file ready to be copy&pasted into overleaf or your favorite LaTex editor near you.
    """
    total_string = ""
    each_lines = []
    if references is None:
        references = ["" for i in range(len(names))]
    if len(references) < len(names):
        references = references + "?"*(len(names)-len(references))
    print(len(names),len(redshifts), len(r50s[:,0]), len(n_sersics[:,0]), len(bt_ratios[:,0]), len(ssmds[:,0]), len(M_stars[:,0]), len(M_bhs), len(references))
    assert len(names) == len(redshifts) == len(r50s[:,0]) == len(n_sersics[:,0]) == len(bt_ratios[:,0]) == len(ssmds[:,0]) == len(M_stars[:,0]) == len(references) == len(M_bhs)
    length = len(names)

    def LatexUncertainty(value):
        val, unc_lo, unc_hi = round(value[0],2), round(value[1],2), round(value[2],2)
        unc_lo, unc_hi = np.max([0.01, unc_lo]), np.max([0.01, unc_hi])
        val, unc_lo, unc_hi = f'{val:.2f}', f'{unc_lo:.2f}', f'{unc_hi:.2f}'
        return f"${val}_{r'{-'+unc_lo+r'}'}^{r'{+'+unc_hi+r'}'}$"

    for i in range(length):
        each_lines.append(names[i]+r"$^{\rm "+references[i]+"}$" + " & " + "$" + str(round(redshifts[i],3)) + "$" + " & " + LatexUncertainty(r50s[i]) + " & " + LatexUncertainty(n_sersics[i]) + " & " + LatexUncertainty(bt_ratios[i]) + " & " + LatexUncertainty(ssmds[i]) + " & " + LatexUncertainty(M_stars[i])+ " & " + LatexUncertainty(M_bhs[i])+ r" \\" + "\n")        

    # swap lines around here easily
    onlyQPEs = np.array(each_lines)[[0,1,2,3,6,7]]
    QPE_TDEs = np.array(each_lines)[[4,5,8]]
    onlyTDEs = np.array(each_lines)[[9,10,11,12,13,14,15,16,17,18,19,20,21]]
    
    total_string += r"\multicolumn{8}{c}{\emph{QPE host galaxies}}\\"+"\n\hline\n\hline\n"
    for line in onlyQPEs:
        total_string += line
        if line != onlyQPEs[-1]:
            total_string += r"\vspace{2pt}"+"\n"
    total_string += "\hline\n"+r"\multicolumn{8}{c}{\emph{TDE+QPE host galaxies}}\\"+"\n\hline\n\hline\n"
    for line in QPE_TDEs:
        total_string += line
        if line != QPE_TDEs[-1]:
            total_string += r"\vspace{2pt}"+"\n"
    total_string += "\hline\n"+r"\multicolumn{8}{c}{\emph{TDE host galaxies}}\\"+"\n\hline\n\hline\n"
    for line in onlyTDEs:
        total_string += line
        if line != onlyTDEs[-1]:
            total_string += r"\vspace{2pt}"+"\n"

    with open(filename, "w") as text_file:
        text_file.write(total_string)
    if verbose:
        print(total_string)




def toLog(a, inverse=False):
    """Convert array of data and uncertainties to/from log base"""
    if not inverse:
        a = np.array(a)
        try:
            data, lo, hi = a[:,0], a[:,1], a[:,2]
        except:
            data, lo, hi = a
        lo = np.abs(lo/(data*np.log(10)))
        hi = np.abs(hi/(data*np.log(10)))
        data = np.log10(data)
        return np.array([data, hi, lo]).T
    else:
        a = np.array(a)
        data, lo, hi = 10**a[:,0], 10**a[:,0]*np.log(10)*a[:,1], 10**a[:,0]*np.log(10)*a[:,2]
        return np.array([data, hi, lo]).T



def SDSS_objid_to_values(objid):
    """Convert an SDSS objID to values"""
    # Determined from http://skyserver.sdss.org/dr7/en/help/docs/algorithm.asp?key=objID

    bin_objid = bin(objid)
    bin_objid = bin_objid[2:len(bin_objid)]
    bin_objid = bin_objid.zfill(64)

    empty = int( '0b' + bin_objid[0], base=0)
    skyVersion = int( '0b' + bin_objid[1:4+1], base=0)
    rerun = int( '0b' + bin_objid[5:15+1], base=0)
    run = int( '0b' + bin_objid[16:31+1], base=0)
    camcol = int( '0b' + bin_objid[32:34+1], base=0)
    firstField = int( '0b' + bin_objid[35+1], base=0)
    field = int( '0b' + bin_objid[36:47+1], base=0)
    object_num = int( '0b' + bin_objid[48:63+1], base=0)

    return skyVersion, rerun, run, camcol, field, object_num

from astropy.coordinates import SkyCoord
import astropy.units as u

# Get the object closest to "pos" in the RAs and DECs lists given
def get_smallest_sep(pos, ras, decs):
    """Get the object closest to "pos" in the RAs and DECs lists given"""
    c1 = SkyCoord(pos[0]*u.deg, pos[1]*u.deg)
    c2 = SkyCoord(ras*u.deg, decs*u.deg)
    sep = (c1.separation(c2)).arcsec
    idx = (sep).argmin()
    return idx, sep[idx]


# Print something in the terminal in a specific color
def print_color(message, color="yellow", **kwargs):
    """print(), but with a color option"""
    possible_colors = ["black","red","green","yellow","blue","magenta","cyan","white"]
    if color == None or color == "grey":
        color = "0"
    elif type(color) == str:
        color = color.lower()
        if color in possible_colors:
            color = str(possible_colors.index(color)+30)
        else:
            print(f"Color '{color}' not implemented, defaulting to grey.\nPossible colors are: {['grey']+possible_colors}")
            color = "0"
    else:
        raise ValueError(f"Parameter 'header_color' needs to be a string.")
    print(f"\x1b[{color}m{message}\x1b[0m", **kwargs)




# Recombine 2D arrays (that were split to save a 3D array) into a single 3D array
def recombine_arrays(data, lo, hi):
    """
    Recombine 2D arrays into a single 3D array
    data: array containing the data
    lo: array containing the lower uncertainties
    hi: array containing the high uncertainties
    """

    assert data.shape == lo.shape == hi.shape

    new_array = np.zeros((*(data.shape),3))
    new_array[:,:,0] = data
    new_array[:,:,1] = lo
    new_array[:,:,2] = hi

    return new_array

# Cut values from catalog if the property at column "index" is not within the bounds given
def cut_from_catalog(catalog, index, bounds, verbose=False):
    """
    catalog: numpy array
    index: int of index of parameter column which is under study here
    bounds: tuple of bound which we want to keep

    returns: numpy array with only remaining objects
    """
    catalog = np.array(catalog)
    if bounds[0] == None:
        good_indices_lo = catalog[:,index] == catalog[:,index]
    else:
        good_indices_lo = catalog[:,index] >= bounds[0]
    if bounds[1] == None:
        good_indices_hi = catalog[:,index] == catalog[:,index]
    else:
        good_indices_hi = catalog[:,index] <= bounds[1]
    good_indices = []
    for i in range(len(good_indices_lo)):
        good_indices.append(good_indices_lo[i] and good_indices_hi[i])
    cut_catalog = catalog[good_indices]
    if verbose:
        print(f"\x1b[31m{catalog.shape[0]-cut_catalog.shape[0]} objects cut\x1b[0m")
        print(f"\x1b[32m{cut_catalog.shape[0]} objects remaining\x1b[0m")
    return cut_catalog

# Add columns of data if object present in both catalogs
def mergeCatalogs_withObjIDs(cat1,cat2,columnsToAdd=[0]):
    """
    Merge catalog1 with catalog2 assuming their first columns are the objIDs
    """
    good_indices = []
    properties_toAdd = []
    for i in range(len(columnsToAdd)):
        properties_toAdd.append([])
    for i in tqdm(range(len(cat1[:,0]))):
        try:
            index = list(cat2[:,0]).index(cat1[i,0])
            good_indices.append(i)
            for k in range(len(columnsToAdd)):
                properties_toAdd[k].append(cat2[index,columnsToAdd[k]])
            #print(f"{cat1[i,0]} vs {cat2[index,0]}")
        except:
            pass
    cat1 = cat1[[good_indices],:][0]
    for i in range(len(columnsToAdd)):
        cat1 = np.vstack((cat1.T, np.array(properties_toAdd[i]))).T
    return cat1

# Cut values from an array using the indices of the values you want to cut
def cut_from_array(a, indices):
    indices = [(i if i >= 0 else len(a)+i) for i in indices]
    return a[[(i not in indices) for i in range(len(a))]]



# Add null (or fixed-value) uncertainties to an array of data
def add_0_uncertainties(a, value=0):
    """
    a: array of shape (n,)
    value: value to set uncertainties (default=0)

    Function to add null uncertainties to an array of data.
    Takes in an array of shape (n,) and returns an array of shape (n,3).
    E.g. 
    [3,4,5,6] -> [[3,0,0],[4,0,0],[5,0,0],[6,0,0]]
    
    """
    a = np.array(a)
    try:
        if a.shape[1] == 3:
            print("No uncertainties added, already containing uncertainties.")
            return a
    except:
        placeholder = np.ones((a.shape[0],3))*value
        placeholder[:,0] = a
        a = placeholder
        return a


# Time your task
import time
if __name__ == "__main__":
    start_time = time.time()
    # Execute task here
    print("\x1b[33m: --- %s seconds ---\x1b[0m" % (time.time() - start_time))




# For SMBHBs

def P_qpo_from_logw0(logw0, logw0_std=None, units="days"):
    """
    Returns the period of the QPO in days
    """
    res = 2*np.pi / np.exp(logw0)
    if units == "seconds":
        res /= 86400
    if logw0_std is None:
        return res
    return np.array([res, res*logw0_std[0], res*logw0_std[1]])

if __name__ == "__main__":
    print(P_qpo_from_logw0(-1.38,[0.01,0.01]))
    # This recovers the period P_qpo = 25 that was given to simulate the lightcurve.


    print(P_qpo_from_logw0(-2.8,[0.0,0.1]))
    print(P_qpo_from_logw0(-13.31,[0.17,0.27])/86400)
    print(P_qpo_from_logw0(-16.91,[0.41,0.30], units="seconds"))

    # Print the period bounds from the fitting (log_d=[-5,5])
    print(P_qpo_from_logw0(-5))
    print(P_qpo_from_logw0(5))