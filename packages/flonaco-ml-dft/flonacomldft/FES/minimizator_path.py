import numpy as np

class Path_:
    def __init__(self, x,y,z, bins=None):
        """ X,Y,Z arrays in 1D eachone
        bins, number of grid points in x axis"""
        self.x_line = x
        self.y_line = y
        self.z_line = z

        if bins == None:
            self.bins = int(np.sqrt(len(self.x_line)))
        else:
            self.bins = bins

        self.x_grid = x.reshape(self.bins,self.bins)
        self.y_grid = y.reshape(self.bins,self.bins)
        self.z_grid = z.reshape(self.bins,self.bins)

        self.grid_points = np.stack((self.x_line, self.y_line), axis=1)

        self.path = np.array([[None]])

        self.neighs = np.array([[0,-1],
                               [1,-1],
                               [1,0],
                               [1,1],
                               [0,1],
                               [-1,1],
                               [-1,0],
                               [-1,-1],
                               [0,0]])


    def path_as_XY(self):
        index_x = self.path[:,0]
        index_y = self.path[:,1]
        assert len(index_x) == len(index_x), "ERROR IN COORDINATES. Different lenghts"

        x = [self.x_grid[i[0]][i[1]] for i in self.path]
        y = [self.y_grid[i[0]][i[1]] for i in self.path]

        return [x,y]

    def fill(self):
        """ this functions guarantees a distance equal to 1
        grid spacing between points of the path """
        i=1
        while i < len(self.path):
            vec1 = self.path[i-1]
            vec2 = self.path[i]
                        
            if (abs(vec1-vec2)>1).any():
                # adds one intermedia point in path when distance is grater 
                #that 1 grid space
                pre = np.round((vec1+vec2)/2).astype(int)
                self.path = np.insert(self.path, i, np.array([pre]), axis=0)
                i-=1
            i += 1
            
    def dots_in_line(self, dots,sep=None):
        """right line that join dots in order from 0 to len_dots[0]
        which has to be the same that len_dots[1]
        
        PARAMETERS
        -----------------
        dots : array with points included in the path. dots are defined as pairs
        of [rows, cols] in the grid matrix
        """
        self.path = dots
        self.fill()
        
        return self.path
    
    
    def ind_to_num(self, pair):
        assert np.all(self.neighs == pair, axis=1).any(), f"error {pair}"
        return np.where(np.all(self.neighs == pair, axis=1))[0][0]

        
    def num_to_ind(self, l):
        assert (l in list(range(9))), f"error searching number of index {l}"
        return self.neighs[l]


    def neighbor(self, num1, num2):
        """ return the neighbor of center excluding num1
        and num2"""
        neis = list(range(9))
        if num1 in neis:
            neis.remove(num1)
        if num2 in neis:
            neis.remove(num2)
        return neis

    def move(self, center, num1, num2):
        """ Choose the minima among the center and the
        neighborhood"""

        neis = self.neighbor(num1, num2)
        
        fe = []
        ns = []
        
        for nei in neis:
            neighbor = center + self.num_to_ind(nei)
            if not self.point_in_traj(neighbor):
                ns.append(neighbor)
            elif (neighbor == center).all():
                ns.append(neighbor)
        [fe.append(self.z_grid[n[0]][n[1]]) for n in ns]
        return ns[fe.index(min(fe))]

    def point_in_traj(self, vec):
        """ This function searches if vec is in trajectory """
        return np.all(self.path == vec, axis=1).any()

    def clean_point(self, i, pos):
        """ This function clean the error as repeated points 
        pos is refered to the previous neighbor (-1) aot the
        posterior (+1)"""
        i += pos
        change = 0
        while change == 0 and i <len(self.path):
            change = 1
            if i-1 > 0 and i+2 <= len(self.path):
                center = self.path[i]
                vec1 = self.path[i-1]
                vec2 = self.path[i+1]
                if (vec1 == center).all() or (vec2 == center).all():
                    #removes adjacent repeated
                    self.path = np.delete(self.path, i, axis=0)
                    change = 0
                  
                else:
                    num1 = self.ind_to_num(vec1-center)
                    num2 = self.ind_to_num(vec2-center)
                    if num1 == num2 or abs(num1-num2) == 1 or abs(num1-num2)==7:
                        #remove this < or this -|
                        self.path = np.delete(self.path, i, axis=0)
                        change = 0
            elif i-1 < 0:
                center = self.path[i]
                vec = self.path[i+1]
                if (vec == center).all():
                    #removes 
                    self.path = np.delete(self.path, i, axis=0)
                    change = 0
            if change == 0 and pos == -1:
                i -= 1
        return i - pos

    def test(self, message=""):
        """ This function probes that the distance between points is lower that 2"""
        dif = self.path[1:] - self.path[:-1]
        i = np.all(abs(dif) < 2, axis=1)
        assert i.all(), "{} large space {} \n {}".format(np.where(i == 0)[0], dif[np.invert(i)], message)

    def clean_recrossing(self):
        values, counts = np.unique(self.path, axis=0, return_counts=True)
        values = values[counts > 1]
        for crossed in values:
            if self.point_in_traj(crossed):
                indexes = np.where(self.path == crossed)[0]
                self.path = np.vstack((self.path[:indexes[0]], self.path[indexes[-1]:]))

    def clean_close(self):
        """This function deletes ns in trajectories, that is |_| """
        i = 0
        while i < len(self.path)-3:
            dif = abs(self.path[i]-self.path[i+3])
            if  (dif == [0,1] ).all() or (dif == [1,0] ).all():
                self.path = np.delete(self.path, [i+1, i+2], axis=0)
            i += 1

    def minimize(self, times=1000):
        """Function that minimizes the initial path"""
        for o in range(times):
            path0 =self.path.copy()

            i = 1
            while i < len(self.path)-1:
                center = self.path[i].copy()
                vec1 = self.path[i-1]
                vec2 = self.path[i+1]

                num1 = self.ind_to_num(vec1-center)
                num2 = self.ind_to_num(vec2-center)
                if num1 == num2 or abs(num1-num2) == 1 or abs(num1-num2)==7:
                    i = self.clean_point(i+1,-1)
                else:
                    
                    mov = self.move(center, num1, num2).copy()
                    self.path[i] = mov
                    if (abs(vec1-mov)>1).any():
                        pre = np.round((vec1+mov)/2).astype(int)
                        self.path = np.insert((self.path[:]),i,np.array([pre]), axis=0)
                        i = self.clean_point(i, -1)
                        i+=1
                    if (abs(vec2-mov)>1).any():
                        pre = ((vec2+mov)/2).astype(int)
                        self.path = np.insert((self.path[:]), i+1, np.array([pre]), axis=0)
                        x='\n'
                        self.clean_point(i+1, 1)
                        i+=1

                    i = self.clean_point(i, -1)
                    self.clean_point(i, 1)
                    
                i += 1
            self.clean_close()

            if np.all(path0 == self.path):
                print("path didn't change")
                break
        traFES=[]
        for traj_point in self.path:
            row, col = traj_point
            traFES.append(self.z_grid[row][col])
        return self.path, traFES
    # ---------------- get index from CV -----------
    def xy_to_index(self, xy):
        x = xy[0]
        y = xy[1]

        dx = self.x_line[1] - self.x_line[0]    #grid spacing in C, dc
        dy = self.y_line[self.bins] - self.y_line[0]  #grid spacing in R, dr

        for i in range(len(self.C_line)):
            if abs(x - self.x_line[i])==approx(0,abs=dx/2)  and abs(y - self.y_line[i])==approx(0,abs=dy/2):
                return i
        raise AssertionError("didn't find the index")

    #------------ get grid from CV ---------------
    def CV_to_grid(self, xy):
        """ Transform from CV to rows,cols in CV_grid """
        i = self.xy_to_index(xy)
        return np.array([i//self.bins, i%self.bins])

    def descendant(self, initial, output='cvs'):
        """ Initial should be the grid coordinates of the initial point """
        old = np.array([0,0])
        new = initial
        length = len(self.x_grid[0])

        i=1

        while np.any(old != new):
            i +=1
            old = new.copy()
            new = self.move(new, 15, 15).copy()
            if any(new == [0,0]) or any(new == [length - 1, length - 1]):
                raise AssertionError("it didn't find a minimum")

        row, col = new
        if output == 'cvs':
            return np.array([self.x_grid[row][col], self.y_grid[row][col]])

        elif output == 'grid':
            return new

        elif output == 'index':
            return int(new[0] * self.bins + new[1])

        raise NameError("Output not recognized")

    # ----------- add new point in climbing
    def add_crest_point(self, border='final'):
        if border == 'final':
            center = self.path[-1]
            vec1 = self.path[-2]
        elif border =='initial':
            center = self.path[0]
            vec1 = self.path[1]
        num1 = self.ind_to_num(vec1 - center)
        return self.move_crest(center, 8, num1)

    def search_local(self, y, maxmin='max'):
        """This function searches all local max (or min) in a function f(x) with x R^1"""
        left = y[1:-1]-y[:-2]
        right = y[1:-1] - y[2:]

        factor = left * right
        factor = np.insert(factor, 0, -1)
        factor = np.append(factor, -1)
        false_left = np.insert(left, 0, -1)
        false_left = np.append(false_left, -1)
        if maxmin == 'max':
            return np.where((factor>0) * (false_left>0))[0]
        elif maxmin == 'min':
            return np.where((factor>0) * (false_left<0))[0]

    def move_crest(self, center, num1, num2):
        """ Choose the minima among the center and the
        neighborhood"""

        neis = [(num2 + i)%8 for i in range(1,8)]
        ns = np.array([center + self.num_to_ind(nei) for nei in neis])
        fe = np.array([-self.z_grid[n[0]][n[1]] for n in ns])
        index = self.search_local(fe)

        if len(index) == 0:
            print("it didn't find a crest point")
            return np.array([-1,-1])
        else:
            n = index[np.where(fe[index] == max(fe[index]))]
            return ns[n]

    # ------------ go climbing

    def climb(self, initial):
        zcopy = self.z_grid.copy()
        path = self.path.copy()

        self.z_grid = -self.z_grid
        self.path = np.array([initial])
        self.path = np.append(self.path, np.array([self.move(initial, 8, 20)]), axis=0)
        point = self.add_crest_point(border='final')
        self.path = np.append(self.path, point, axis=0)

        while (np.linalg.norm(self.path[-1]-self.path[0]) > 1.5) and \
              (self.path[-1] != [self.bins-1,self.bins-1]).all() and \
              (self.path[-1] != [0,0]).all():

              point = self.add_crest_point(border='final')
              
              if not (point == -1).all():
                  self.path = np.append(self.path, point, axis=0)
              else:
                  print('break')
                  break
              if not (abs(self.path[-1]-self.path[0]) != [1,1]).all():
                  print("cola toca punta")
              elif not (self.path[-1] != [self.bins-1,self.bins-1]).all():
                  print("cola toca borde máximo")
              elif not (self.path[-1] != [0,0]).all():
                  print("cola toca borde mínimo")  

        while (np.linalg.norm(self.path[-1]-self.path[0]) > 1.5) and \
              (self.path[0] != [self.bins-1,self.bins-1]).all() and \
              (self.path[0] != [0,0]).all():
              point = self.add_crest_point(border='initial')
              if not (point == -1).all():
                  self.path = np.insert(self.path, 0, point, axis=0)
              else:
                  print('break')
                  break

              if not (abs(self.path[-1]-self.path[0]) != [1,1]).all():
                  print("cola toca punta")
              elif not (self.path[0] != [self.bins-1,self.bins-1]).all():
                  print("cola toca borde máximo")
              elif not (self.path[0] != [0,0]).all():
                  print("cola toca borde mínimo")  



        return self.path




        

        

        
        
