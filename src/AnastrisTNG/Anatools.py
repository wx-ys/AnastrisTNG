import numpy as np
import scipy



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
            coef=scipy.linalg.solve(A,B)
            self.coefx.append(coef)      
                  
            x1=self.y[i]
            x2=self.y[i+1]
            v1=self.vy[i]
            v2=self.vy[i+1]
            A=np.array([[1,t1,t1**2,t1**3],[1,t2,t2**2,t2**3],[0,1,2*t1,3*t1**2],[0,1,2*t2,3*t2**2]])
            B=np.array([x1,x2,v1,v2])
            coef=scipy.linalg.solve(A,B)
            self.coefy.append(coef)      
            
            x1=self.z[i]
            x2=self.z[i+1]
            v1=self.vz[i]
            v2=self.vz[i+1]
            A=np.array([[1,t1,t1**2,t1**3],[1,t2,t2**2,t2**3],[0,1,2*t1,3*t1**2],[0,1,2*t2,3*t2**2]])
            B=np.array([x1,x2,v1,v2])
            coef=scipy.linalg.solve(A,B)
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
    