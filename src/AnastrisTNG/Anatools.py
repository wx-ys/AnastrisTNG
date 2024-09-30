'''
Some useful functions
orbit: fit the orbit, based on the observed pos,vel,t
ang_mom: vector
angle_between_vectors:
fit_krotmax:

'''

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from pynbody.analysis.angmom import calc_faceon_matrix


class orbit():
    def __init__(self,pos,vel,t):
        '''
        input:
            pos, shape: Nx3, x,y,z
            vel, shape: Nx3, vx,vy,vz
            T, shape: 1xN, (t1,t2,t3...tn)
            sliceT, int or array, 
        output:
            pos,
            vel
            T,
        #TODO a, defalt: None, ax,ay,az
        '''
        t=np.asarray(t)
        Targ=np.argsort(t)
        pos=np.asarray(pos)[Targ]
        vel=np.asarray(vel)[Targ]
        t=t[Targ]
        self.pos=pos
        self.vel=vel
        self.t=t
        self.x=pos.T[0]
        self.y=pos.T[1]
        self.z=pos.T[2]
        self.vx=vel.T[0]
        self.vy=vel.T[1]
        self.vz=vel.T[2]
        self._fit()
        self.tmax=np.max(self.t)
        self.tmin=np.min(self.t)
        
    def _fit(self):
        self.coefx=[]
        self.coefy=[]
        self.coefz=[]
        for i in range(len(self.t)-1):
            t1=self.t[i]
            t2=self.t[i+1]
            
            x1=self.x[i]
            x2=self.x[i+1]
            v1=self.vx[i]
            v2=self.vx[i+1]
            A=np.array([[1,t1,t1**2,t1**3],[1,t2,t2**2,t2**3],[0,1,2*t1,3*t1**2],[0,1,2*t2,3*t2**2]])
            B=np.array([x1,x2,v1,v2])
            coef=solve(A,B)
            self.coefx.append(coef)      
                  
            x1=self.y[i]
            x2=self.y[i+1]
            v1=self.vy[i]
            v2=self.vy[i+1]
            A=np.array([[1,t1,t1**2,t1**3],[1,t2,t2**2,t2**3],[0,1,2*t1,3*t1**2],[0,1,2*t2,3*t2**2]])
            B=np.array([x1,x2,v1,v2])
            coef=solve(A,B)
            self.coefy.append(coef)      
            
            x1=self.z[i]
            x2=self.z[i+1]
            v1=self.vz[i]
            v2=self.vz[i+1]
            A=np.array([[1,t1,t1**2,t1**3],[1,t2,t2**2,t2**3],[0,1,2*t1,3*t1**2],[0,1,2*t2,3*t2**2]])
            B=np.array([x1,x2,v1,v2])
            coef=solve(A,B)
            self.coefz.append(coef)   
        self.coefx=np.array(self.coefx)
        self.coefy=np.array(self.coefy)
        self.coefz=np.array(self.coefz)

               
    def get(self,t):
        if not hasattr(t,'__iter__'):
            t=[t]
        t=np.array(t)

        ti=np.searchsorted(self.t,t,side='left')-1
        ti[t<self.tmin]=0
        ti[t>self.tmax]=0
        ti[ti==len(self.t)-1]=len(self.t)-2

        POS=np.array([self.coefx[ti].T[0]+self.coefx[ti].T[1]*t+self.coefx[ti].T[2]*t**2+self.coefx[ti].T[3]*t**3,
                      self.coefy[ti].T[0]+self.coefy[ti].T[1]*t+self.coefy[ti].T[2]*t**2+self.coefy[ti].T[3]*t**3,
                      self.coefz[ti].T[0]+self.coefz[ti].T[1]*t+self.coefz[ti].T[2]*t**2+self.coefz[ti].T[3]*t**3
                      ]).T

        VEL=np.array([self.coefx[ti].T[1]+2*self.coefx[ti].T[2]*t+3*self.coefx[ti].T[3]*t**2,
                      self.coefy[ti].T[1]+2*self.coefy[ti].T[2]*t+3*self.coefy[ti].T[3]*t**2,
                      self.coefz[ti].T[1]+2*self.coefz[ti].T[2]*t+3*self.coefz[ti].T[3]*t**2
                      ]).T
        POS[t<self.tmin]=np.array([0,0,0])
        POS[t>self.tmax]=np.array([0,0,0])
        VEL[t<self.tmin]=np.array([0,0,0])
        VEL[t>self.tmax]=np.array([0,0,0])
        return POS,VEL
    
# from pynbody.analysis.angmom.ang_mom_vec    
def ang_mom(snap):
    angmom = (snap['mass'].reshape((len(snap), 1)) *
              np.cross(snap['pos'], snap['vel'])).sum(axis=0).view(np.ndarray)
    return angmom


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    cos_theta = dot_product / (v1_norm * v2_norm)
    theta=np.arccos(cos_theta)
    return np.degrees(theta)

def get_krot(pos,vel,mass,Rota):
    pos_new = np.dot(Rota, pos.T)
    vel_new = np.dot(Rota, vel.T)
    rxy=np.sqrt((pos_new[0]**2 +pos_new[1] ** 2))
    vcxy=(pos_new[0] * vel_new[1] - pos_new[1]  * vel_new[0]) / rxy
    Krot = np.array(np.sum((0.5*mass*(vcxy** 2))) / np.sum(mass*0.5 * (vel_new ** 2).sum(axis=0)))  
    return Krot

def _kroterr(initpa, *args):
    
    x, y, z = initpa
    if (x+y+z) == 0:
        z = 1
    Rc = np.array([x, y, z])
    Rota = calc_faceon_matrix(Rc)
    pos,vel,mass = args
    Krot = get_krot(pos,vel,mass,Rota)
    return 100-Krot*100 

def fit_krotmax(pos,vel,mass, method='BFGS'):
    
    res = minimize(_kroterr,
                (0,0.,1.),
                (pos,vel,mass),
                method=method,
               )
    if res.success:
        result = {
            'krotmax': (1-res.fun/100),
            'krotvec': (res.x/np.linalg.norm(res.x)),
            'krotmat': calc_faceon_matrix(res.x),
            'angle': angle_between_vectors(res.x, np.array([0,0,1]))
        }
        return result
    else:
        print('Failed to fit maximum krot')
        return None