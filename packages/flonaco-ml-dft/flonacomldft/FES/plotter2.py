import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
import glob
from subprocess import Popen
import cmocean.cm as cmo
import matplotlib as mpl
import sys
from pathlib import Path

from flonacomldft.FES.minimizator_path import Path_
from flonacomldft.utils.io_utils import get_path
from pytest import approx

# TODO:
class Plotter():
    def __init__(self, bins, cluster, shift=0):
        bins += 1
        self.bins = bins
        self.shift = shift
        self.silver_minima = cvs_points()[cluster] #in space, non-roted
        self.neighs = np.array([[0,-1],
                               [1,-1],
                               [1,0],
                               [1,1],
                               [0,1],
                               [-1,1],
                               [-1,0],
                               [-1,-1],
                               [0,0]])

    def generate_files(self,dire, out, name='fes.dat'):
        """ This function takes all the feses the sub-directories of 'dire' 
        called 'name' and computes the mean and error in each gridpoint. The
        result is saved in dire/out file"""

        a = glob.glob(dire+"/*/{}".format(name))
        C = np.loadtxt(a[0], usecols=0)
        R = np.loadtxt(a[0], usecols=1)
        m = np.loadtxt(a[0], usecols=2)
        v = (np.loadtxt(a[0],usecols=2))**2

        i=1
        for filen in a[1:]:
            m += np.loadtxt(filen, usecols=2)
            v += np.loadtxt(filen, usecols=2)**2
            i += 1

        F = m/i
        F2 = v/i

        var = np.sqrt((F2-F**2)/i)
        with open(dire+'/'+out, 'w') as fi:
            fi.write('# C, R, FE, var\n')
            for i in range(len(C)):
                fi.write('{} \t {} \t {} \t {}\n'.format(C[i], R[i], F[i], var[i]))

        self.C_line = C
        self.R_line = R
        self.F_line = F
        self.V_line = var

        self.C_grid = C.reshape(self.bins, self.bins)
        self.R_grid = R.reshape(self.bins, self.bins)
        self.F_grid = F.reshape(self.bins, self.bins)
        self.V_grid = var.reshape(self.bins, self.bins)


    def readfile(self, fil, rot=False, vari=False):
        """ This reads the fes in fil and save the values of the grid points
        coordinates, and the fes. In case of vari, it saves the error too in
        the internal variables"""

        self.C_line = np.loadtxt(fil, usecols=0)
        self.R_line = np.loadtxt(fil, usecols=1)
        self.F_line = np.loadtxt(fil, usecols=2)
        
        self.C_grid = self.C_line.reshape(self.bins, self.bins)
        self.R_grid = self.R_line.reshape(self.bins, self.bins)
        self.F_grid = self.F_line.reshape(self.bins, self.bins)

        if vari:
            self.V_line = np.loadtxt(fil,usecols=3)
            self.V_grid = [self.V_line[i*self.bins:(i+1)*self.bins] for i in range(len(self.V_line)//self.bins)]
    
    def rot(self,c=None, r=None):
        """This funcion apply the rotation to C and R,
        giving the values of CV1 and CV2"""
        if c is None and r is None:
            c = self.C_grid
            r = self.R_grid
        C = 0.99715 * c - 0.07534 * r
        R = 0.07534 * c + 0.99715 * r
        return [C,R]

    def unrot(self, c=None, r=None):
        """This funcion apply the inverse rotation to CV1 and CV2,
        giving the correct values of C and R"""
        if c is None and r is None:
            c = self.C_grid
            r = self.R_grid
        C = 0.99715 * c + 0.07534 * r
        R = -0.07534 * c + 0.99715 * r
        return [C,R]

    def add_points(self, ax, labels=40, ticks=25, size1=8, arrows=False, rot=False):
        """ This function adds the points of the minima"""
        
        minima = self.silver_minima.copy()
        if rot:
            for i,minimum in enumerate(minima):
                minima[i] = self.rot(minimum[0], minimum[1]) + [minimum[2]]
        else:
            for i,minimum in enumerate(minima):
                minima[i] = [minimum[0], minimum[1], [minimum[2]]]
        size2 = size1*0.6
        for minimum in minima:
            ax.plot(minimum[0],minimum[1], 'o', markersize=size1,color = 'white')
            ax.plot(minimum[0],minimum[1], '*', markersize=size2,color = 'red', label=minimum[2])

        l = [mpe.Stroke(foreground='white')]

        if arrows:
            if type(arrows) == list:
                factor = arrows
            else:
                factor = [arrows] * len(minima)
            for i, minimum in enumerate(minima):
                if i < len(factor):
                    ax.annotate(str(i+1), xy=minimum[:-1], xytext=np.array(minimum[:-1])+np.array([0.1,0.01])*factor[i], 
                                arrowprops=dict(arrowstyle="-|>, head_length=0.3,  head_width=0.15",
                                                mutation_scale=40, fc='black', linewidth=3), 
                                fontsize=labels)


    def plotter(self, delta, temp, F, labbar, deci=1, bar=True, lim=0,
                unrot=False, rot=False, rot_points=False,
                labels=40, ticks=25, arrows=None,
                add_paths=0, shift=0, cmap=cmo.tempo_r,
                ax=None, fig=None, minmax=[None,None],
                orientation='vertical', save=0, labelaxis=None,
                delta2=None, isopotentials=None):
        """ This functions plots a 3d map

        PARAMS
        -----------------

        delta: difference between curves lines

        temp: temperature

        F: z component in grid structure

        labbar: label bar

        deci (1): number of decimals in the bar

        bar (True): if the bar is included in the graph

        lim (automatic): range of values of the plot

        unrot (False): inverse rotation of coordinates

        rot (False): rotation of coordinates

        labels (40): size of labels

        ticks (25): size of ticks

        arrows (0): lenght of arrows pointing minima. 0 if not arrows but points

        add_paths (0): paths to be added to the FES in cvs. n-paths in
                       [xs,ys] structure each one.
        
        shift (0): shift the FES.

        cmap (cmo.tempo_r): color map for the FES

        ax (None): axis to plotting. In case of none, it is created.

        fig (None): Figure to plotting. In case of none, it is created.

        minmax ([None,None]): list with the min,max values of the FES bar

        orientation ('vertical'): string with orientation of the bar

        save(0): 1 if the plot will be saved

        labelaxis(None): label of the axis in a list with labelx,labely structure
        """

        if unrot:
            C,R = self.unrot()
        elif rot:
            C,R = self.rot()
        else:
            C = self.C_grid
            R = self.R_grid
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(13,10))
        ax.tick_params(axis='y', labelsize=ticks)
        ax.tick_params(axis='x', labelsize=ticks)
        
        if arrows is not None:
            self.add_points(ax, ticks=ticks, arrows=arrows, rot=rot_points)
        
        mini = 100
        maxi = -100
        
        if minmax[0] is None:
            for fesvalue in F:
                if fesvalue is not None:
                    if fesvalue < mini:
                        mini = fesvalue
                    if fesvalue > maxi:
                        maxi = fesvalue
            if labbar[0]=="F":
                maxi += 0
            else:
                maxi += delta
        else:
            mini = minmax[0]
            maxi = minmax[1]

        F_grid = F.reshape((self.bins, self.bins))
        print(type(minmax))

        im = ax.contourf(C.astype(float), R.astype(float), F_grid.astype(float), np.arange(mini,maxi,delta), cmap=cmap, 
                         vmin=minmax[0], vmax=minmax[1])
        if delta2 is None:
            delta2 = delta
        cp = ax.contour(C.astype(float), R.astype(float), F_grid.astype(float), np.arange(mini,maxi,delta), 
                         linestyles='-', colors = 'darkgray', linewidths=1.2)
        if isopotentials is not None:
            ax.contour(C.astype(float), R.astype(float), F_grid.astype(float), isopotentials, 
                         linestyles='-', colors = 'red', linewidths=1.5)
        if orientation[0]=='v':
            pad = 0.02
            shrink = 1
            rotation=0
        else:
            pad = 0.15
            shrink = 0.9
            rotation=90
            
        if bar:
            cbar=fig.colorbar(im, ax = ax, format='%1.{}f'.format(deci),orientation=orientation, pad=pad, shrink=shrink)
            #cbar.set_label(label = labbar, fontsize=labels)
            cbar.set_ticks(np.arange(mini,maxi,delta2))
            #cbar.ax.tick_params(labelsize=(ticks+labels)/2, rotation=rotation)

        if lim:
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])
        if labelaxis is None:
            ax.set_xlabel('Coordination Number', fontsize=labels)
            ax.set_ylabel(r'Radius of Gyration [$\mathrm{\AA}$]', fontsize=labels)
        else:
            ax.set_xlabel(labelaxis[0], fontsize=labels)
            ax.set_ylabel(labelaxis[1], fontsize=labels)

        
        self.xs = ax.get_xlim()
        self.ys = ax.get_ylim()
        
        if add_paths:
            for i, min_path in enumerate(add_paths):
                ax.plot(min_path[0], min_path[1], '*', label=f"{i}")
                plt.legend()

        if temp:
            k = 0.95
            posx = (1-k)*self.xs[0] + k*self.xs[1]
            posy = (1-k)*self.ys[0] + k*self.ys[1]
            #ax.text(posx, posy, '{}K'.format(temp), fontsize=labels*1.2, fontweight='bold', ha='right', va='top')
        
        if save:
            plt.savefig(save)
        
        return fig, ax
    
    
    def plot_fes(self, delta, temp, shift=0, barlabel=None, **kwargs):
        """function that plots the free energy surface"""
        F = self.F_line + shift
        F = np.array([(f if f<=0 else None) for f in F])

        if barlabel is None:
            barlabel = 'eV'
        fig, ax = self.plotter(delta, temp, F, 'FES [{}]'.format(barlabel),**kwargs)

        return fig, ax

    def plot_error(self, delta, temp, shift=0, barlabel=None, **kwargs):
        """function that plots the error in the FE surface"""
        F = (self.F_line + shift)
        F = [(f if f<=0 else None) for f in F]
        V = np.array([(self.V_line[i] if F[i] is not None else None) for i in range(len(self.V_line))])

        if barlabel is None:
            barlabel = 'eV'

        ax = self.plotter(delta, temp, V, 'Error [{}]'.format(barlabel), cmap=cmo.turbid,  **kwargs)

        return ax

    # ----------- Probability analysis ---------------------
    def probas(self, T, shift=0, sep=None, graph=False, **kwargs):
        """functions that computes the probability of each state
        accoding to the free energy surface"""
        if sep == None:
            if self.bins == 201:
                sep = 8643
            if self.bins == 401:
                sep = 34486

        F = self.F_line + shift
        F = [(f if f<=0 else None) for f in F]
        C = self.C_line
        R = self.R_line
        FsA = F[sep:]
        FsB = F[:sep]

        fesA = np.array([x for x in FsA if x is not None]) 
        fesB = np.array([x for x in FsB if x is not None])

        kT = 8.617333e-5 * T
        dC = C[1]-C[0]    #grid spacing in C, dc
        dR = R[self.bins]-R[0]  #grid spacing in R, dr

        #eq 3.16
        expA = np.exp(-fesA/kT)  
        expB = np.exp(-fesB/kT)  

        # Integral over the staes
        stateA = np.sum(expA)*dC*dR
        stateB = np.sum(expB)*dC*dR
        print((f'$P_A$ = $\int_A$ exp[-$\\beta$ F(s)] ds = {stateA}'))
        print((f'$P_B$ = $\int_B$ exp[-$\\beta$ F(s)] ds = {stateB}'))
        print((f'$P_B$/$P_A$ = {stateB/stateA}'))
        print('min_B - min_A =', min(fesB)-min(fesA))
        
        if graph:

            lenA = len(FsA)
            lenB = len(FsB)

            FsA = [None]*lenB + FsA
            FsB = FsB + [None]*lenA

            fig, axes = plt.subplots(1,3, figsize = (25,10))

            self.plotter(0.01, T, np.array(F), 'FES [eV]', ax=axes[0], **kwargs)
            self.plotter(0.01, T, np.array(FsA), 'FES [eV]', ax=axes[1], **kwargs)
            self.plotter(0.01, T, np.array(FsB), 'FES [eV]', ax=axes[2], **kwargs)


            return axes
        else:
            return stateA,stateB
        

    #------------ get index from CV ---------------

    def CV_to_index(self, cvs, rot):
        """ Transform from CV to index in CV_line """
        if rot:
            cvs = self.rot(cvs[0], cvs[1])
        C = cvs[0]
        R = cvs[1]

        dC = self.C_line[1] - self.C_line[0]    #grid spacing in C, dc
        dR = self.R_line[self.bins]-self.R_line[0]  #grid spacing in R, dr

        for i in range(len(self.C_line)):
            if abs(C - self.C_line[i])==approx(0,abs=dC/2)  and abs(R - self.R_line[i])==approx(0,abs=dR/2):
                return i
        raise AssertionError("didn't find the index")

    #------------ get grid from CV ---------------

    def CV_to_grid(self, cvs, rot):
        """ Transform from CV to rows,cols in CV_grid """
        i = self.CV_to_index(cvs,rot)
        return np.array([i//self.bins, i%self.bins])

    
    # -------------- Minimize_all ------

    #def get_minima_dot(self, dot):

    def get_minima_FES(self, isomers=None, rot_points=False, output='cvs'):
        """ Return the minima in the FES """
        if isomers is None:
            isomers = [i[-1] for i in self.silver_minima]
        
        minima = {}

        p = Path(self.C_line, self.R_line, self.F_line)

        for isomer in isomers:
            for coordinate in self.silver_minima:
                if coordinate[-1] == isomer:
                    grid_coord = self.CV_to_grid(coordinate[:-2], rot_points)
                    minima[coordinate[-1]] = p.descendant(grid_coord, output='grid')
                    a = minima[coordinate[-1]]
                    minima[coordinate[-1]] = np.append(a, self.F_grid[a[0]][a[1]])
        if output == 'cvs':
            for minimum in minima.keys():
                grid = minima[minimum].copy()[:-1].astype(int)
                minima[minimum][0] =  self.C_grid[grid[0]][grid[1]]
                minima[minimum][1] =  self.R_grid[grid[0]][grid[1]]
        elif output == 'index':
            for minimum in minima.keys():
                grid = minima[minimum].copy()[:-1].astype(int)
                minima[minimum] =  grid[0] * self.bins + grid[1]
        return minima

    # ----------- Minima of each reconstructed FES ---------------
    def dots_in_feses(self, directories, name_fes, rot_dots, temp, isomers, sep=None, dis=2):
        states = [self.CV_to_grid(self.silver_minima[isomer][:-1], rot_dots) for isomer in isomers]

        mpl.rcParams["errorbar.capsize"] = 3
        
        C = self.C_line
        R = self.R_line

        # ------- set of feses -----

        samples = glob.glob(directories)

        indexes = {}
        for isomer in range(len(states)):
            indexes[isomer] = np.array([])

        for i,dire in enumerate(samples):
            fes = np.loadtxt(dire+f'/{name_fes}', usecols=2)
            p = Path(self.C_line, self.R_line, fes)
            for j,state in enumerate(states):
                indexes[j] = np.append(indexes[j],p.descendant(state, output='index')).astype(int)

        return indexes
    
    # ------------------- convergence -----------
    
    def converAB(self, dire, temp, isomers, sep=None, dis=2, order=1, name='fes.dat', rot=False):
        """ Find the mean and the standard error using n random feses until n = number
        of trajectories """
        stateA = self.CV_to_grid(self.silver_minima[isomers[0]][:-1], rot)
        stateB = self.CV_to_grid(self.silver_minima[isomers[1]][:-1], rot)

        mpl.rcParams["errorbar.capsize"] = 3
        a = glob.glob(dire+"/*/{}".format(name))
        m = np.loadtxt(a[0], usecols=2)
        m1 = np.loadtxt(a[0], usecols=2)*0
        v1 = (np.loadtxt(a[0],usecols=2))*0

        C = self.C_line
        R = self.R_line

        diff = []
        difmean = []
        dots = [[],[]]
        error = []
        i=0
        for filen in a[::order]:
            print(filen)
            i += 1
            m = np.loadtxt(filen, usecols=2)
            m1 += m
            v1 += m**2
            p = Path(C,R,m1)
            index_a = p.descendant(stateA, output='index')
            index_b = p.descendant(stateB, output='index')

            difmean.append((m1[index_b]-m1[index_a])/i)
            
            #error:
            error_a = np.sqrt((v1[index_a]/i - (m1[index_a]/i)**2)/i)
            error_b = np.sqrt((v1[index_b]/i - (m1[index_b]/i)**2)/i)
            error.append(np.sqrt(error_a**2 + error_b**2))

            dots[0].append([C[index_a], R[index_a]])
            dots[1].append([C[index_b], R[index_b]])
        
        x = list(range(1,len(difmean)+1))
        plt.figure(figsize=(10,7))
        plt.errorbar(x, difmean, yerr=error, ls='dashed', marker='o')
        plt.xticks(x[::dis], fontsize = 20);
        plt.yticks(fontsize = 20);

        xs = plt.xlim()
        ys = plt.ylim()

        k = 0.95
        posx = (1-k)*xs[0] + k*xs[1]
        posy = (1-k)*ys[0] + k*ys[1]
        plt.text(posx, posy, '{}K'.format(temp), fontsize=40, fontweight='bold', ha='right', va='top')
        
        plt.xlabel('N', fontsize = 40)
        plt.ylabel('$E_B$ - $E_A$', fontsize = 40) 
        plt.tight_layout(pad=2)
        plt.show(block=False)
        return [difmean, error, dots]

    # ---------- bootstrapping ---------------------

    def set_minima_grid(self, isomers, rot):
        self.stateA = self.CV_to_grid(self.silver_minima[isomers[0]][:-1], rot)
        self.stateB = self.CV_to_grid(self.silver_minima[isomers[1]][:-1], rot)
        return self.stateA, self.stateB

    def feses(self, direction, name='fes.dat'):
        bins = self.bins**2
        samples = glob.glob(direction)
        feses = np.zeros((len(samples), bins))
        
        for i,dire in enumerate(samples):
            feses[i] = np.loadtxt(dire+f'/{name}', usecols=2)
            
        return feses
            

    def extract_differences(self, feses, states):
        differences = []
        for fes in feses:
            p = Path(self.C_line, self.R_line, fes)
            mA = p.descendant(states[0], output='index')
            mB = p.descendant(states[1], output='index')
            diff = fes[mB] - fes[mA]
            differences.append(diff)

        return differences

    def bootstrap_shuffle(self, feses,n,m):
        """ 
        n: number of elements per subset
        m: number of subsets (means)
        """

        bins = self.bins**2
        
        meanx = np.zeros((m, bins))

        for j in range(m):
            rand = np.random.randint(len(feses), size=n)
            iensamble = np.zeros(bins)
            for i in range(n):
                iensamble += feses[rand[i]]
            meanx[j] = iensamble/n
            
        return meanx

    def conver_bootstrap_AB(self, directories, name_fes, rot_dots, temp, isomers, sep=None, dis=2):
            stateA = self.CV_to_grid(self.silver_minima[isomers[0]][:-1], rot_dots)
            stateB = self.CV_to_grid(self.silver_minima[isomers[1]][:-1], rot_dots)
            states = [stateA, stateB]

            mpl.rcParams["errorbar.capsize"] = 3
            
            C = self.C_line
            R = self.R_line

            # ------- set of feses -----

            samples = glob.glob(directories)
            feses = np.zeros((len(samples), len(C)))
            
            for i,dire in enumerate(samples):
                print(dire)
                feses[i] = np.loadtxt(dire+f'/{name_fes}', usecols=2)
            

            difmean = []
            error = []
            
            for n in range(1,len(samples)+1):
                # get 50 feses averaged from n random samples of feses set:
                sub_feses = self.bootstrap_shuffle(feses, n, 50) 
                # Then, 
                diff = self.extract_differences(sub_feses, states)
                difmean.append(np.mean(diff))
                
                #error:
                error.append(np.std(diff))
            
            x = list(range(1,len(difmean)+1))
            plt.figure(figsize=(10,7))
            plt.errorbar(x, difmean, yerr=error, ls='dashed', marker='o')
            plt.xticks(x[::dis], fontsize = 20);
            plt.yticks(fontsize = 20);
            xs = plt.xlim()
            ys = plt.ylim()

            k = 0.95
            posx = (1-k)*xs[0] + k*xs[1]
            posy = (1-k)*ys[0] + k*ys[1]
            plt.text(posx, posy, '{}K'.format(temp), fontsize=40, fontweight='bold', ha='right', va='top')
            plt.xlabel('Number of trajectories N', fontsize = 40)
            plt.ylabel('FE Differences', fontsize = 40) 
            plt.tight_layout(pad=2)
            plt.show(block=False)
            return [difmean, error]

    def dont_colapse(self):
        plt.show()

'''

    def converT(self, dire, temp, py, sep=None, dis=2, order=1, name='fes.dat'):
        if sep == None:
            if self.bins == 201:
                sep = 8643
            if self.bins == 401:
                sep = 34486
        mpl.rcParams["errorbar.capsize"] = 3
        a = glob.glob(dire+"/*/{}".format(name))
        m = np.loadtxt(a[0], usecols=2)
        m1 = np.loadtxt(a[0], usecols=2)*0
        v1 = (np.loadtxt(a[0],usecols=2))*0

        C = self.C_line
        R = self.R_line

        diff = []
        minA = []
        minB = []
        difmean = []
        dots = [[],[]]
        error = []
        i=0
        for filen in a[::order]:
            i += 1
            self.readfile(filen)
            m = self.F_line
            m1 += m
            v1 += m**2
            mA = m[sep:]
            mB = m[:sep]
            m1A = m1[sep:]
            m1B = m1[:sep]

            self.rightline()
            _, _, fesTra = self.minimize()

            index_a = np.where(m1 == min(m1A))[0][0]
            index_b = np.where(m1 == min(m1B))[0][0]

            difmean.append((m1[index_b]-m1[index_a])/i)
            
            #error:
            error_a = np.sqrt((v1[index_a]/i - (m1[index_a]/i)**2)/i)
            error_b = np.sqrt((v1[index_b]/i - (m1[index_b]/i)**2)/i)
            error.append(np.sqrt(error_a**2 + error_b**2))

            dots[0].append([C[index_a], R[index_a]])
            dots[1].append([C[index_b], R[index_b]])
        x = list(range(1,len(difmean)+1))
        plt.figure(figsize=(10,7))
        plt.errorbar(x, difmean, yerr=error, ls='dashed', marker='o')
        plt.xticks(x[::dis], fontsize = 20);
        plt.yticks(fontsize = 20);
        plt.text(8.3, py, 'T={}K'.format(temp), fontsize=40)
        plt.xlabel('N', fontsize = 40)
        plt.ylabel('$E_B$ - $E_A$', fontsize = 40) 
        plt.tight_layout(pad=2)
        plt.show(block=False)
        return [difmean, error, dots]
    
    def several_paths(self, dire, sep=None, dis=2, order=1, name='fes.dat'):
        if sep == None:
            if self.bins == 201:
                sep = 8643
            if self.bins == 401:
                sep = 34486
        mpl.rcParams["errorbar.capsize"] = 3
        a = glob.glob(dire+"/*/{}".format(name))
        

        C = self.C_line
        R = self.R_line

        i=0
        paths = []
        
        for filen in a[::order]:
            i += 1
            self.readfile(filen)
            self.rightline()
            rows, cols, fesTra = self.minimize(100)

            paths.append([rows, cols])

        return paths
'''


class CVs:
    def __init__(self):
        self.time = None
        self.C = None
        self.R = None
        
    def readfile(self, file):
        self.time = np.loadtxt(file, usecols=0)
        self.C = np.loadtxt(file, usecols=1)
        self.R = np.loadtxt(file, usecols=2)
        
        return self.C, self.R
    
    def rot(self):
        C = 0.99715 * self.C - 0.07534 * self.R
        R = 0.07534 * self.C + 0.99715 * self.R
        return [C,R]
    
    def unrot(self):
        C = 0.99715 * self.C + 0.07534 * self.R
        R = -0.07534 * self.C + 0.99715 * self.R
        return [C,R]
    
    def plotter(self, coord, deci=1, bar=True, lim=0, cur=5, size=7,
                func='pval', plot=0, rot=False, 
                 labels=40, ticks=25, arrows=False,
                 add_path=0, add_paths=0,shift=0, cmap=cmo.tempo_r,
                 tpos=[10.8, 2.5], ax=None, fig=None, minmax=[None,None],
                 orientation='vertical'):

        if rot:
            CR = self.rot()
        else:
            CR = [self.C, self.R]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(13,10))
        ax.tick_params(axis='y', labelsize=ticks)
        ax.tick_params(axis='x', labelsize=ticks)
        
        ax.plot(self.time, CR[coord], lw=2)
        
        
        if lim:
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])
        else:
            ax.set_xlim([0, len(CR[0])])
            
        
        axis_labels = ['CV1', r'CV2']
        ax.set_xlabel('time', fontsize=labels)
        ax.set_ylabel(axis_labels[coord], fontsize=labels)
        
        self.xs = ax.get_xlim()
        self.ys = ax.get_ylim()


# from functions.py
def cvs_points(): 
    if 'marylou' in get_path():
        path_data_points = get_path() 
    elif os.path.isdir(str(Path.home())+'/flonaco-ml-dft/flonacomldft/FES/'):
        path_data_points = str(Path.home())+'/flonaco-ml-dft/flonacomldft/FES/'
    elif os.path.isdir(str(Path.home())+'/ML-DFT/packages/flonaco-ml-dft/flonacomldft/FES/'):
        path_data_points = str(Path.home())+'/ML-DFT/packages/flonaco-ml-dft/flonacomldft/FES/'
    elif os.path.isdir(str(Path.home())+'/ceph/ML-DFT/packages/flonaco-ml-dft/flonacomldft/FES/'):
        path_data_points = str(Path.home())+'/ceph/ML-DFT/packages/flonaco-ml-dft/flonacomldft/FES/'
    else:
        raise RuntimeError('Path to points_cvsLCAO.dat not understood')
    
    infoa = np.loadtxt(path_data_points+'points_cvsLCAO.dat', unpack=True, usecols=[0,1], dtype=np.str)
    infob = np.loadtxt(path_data_points+'points_cvsLCAO.dat', unpack=True, usecols=[2,3,4])
    info_dic = {}
    for cluster in infoa[0]:
        info_dic[cluster]=[]
    for i,cluster in enumerate(infoa[0]):
        info_dic[cluster].append([infob[0][i], infob[1][i], infob[2][i], infoa[1][i]])
    return info_dic
