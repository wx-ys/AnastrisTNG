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
        self.__pos=pos
        self.__vel=vel
        self.__t=t
        self.__x=pos.T[0]
        self.__y=pos.T[1]
        self.__z=pos.T[2]
        self.__vx=vel.T[0]
        self.__vy=vel.T[1]
        self.__vz=vel.T[2]
        self.__fit()
        self.tmax=np.max(self.__t)
        self.tmin=np.min(self.__t)
        
    def __fit(self):
        self.__coefx=[]
        self.__coefy=[]
        self.__coefz=[]
        for i in range(len(self.__t)-1):
            coefti=[]
            t1=self.__t[i]
            t2=self.__t[i+1]
            
            x1=self.__x[i]
            x2=self.__x[i+1]
            v1=self.__vx[i]
            v2=self.__vx[i+1]
            A=np.array([[1,t1,t1**2,t1**3],[1,t2,t2**2,t2**3],[0,1,2*t1,3*t1**2],[0,1,2*t2,3*t2**2]])
            B=np.array([x1,x2,v1,v2])
            coef=scipy.linalg.solve(A,B)
            self.__coefx.append(coef)      
                  
            x1=self.__y[i]
            x2=self.__y[i+1]
            v1=self.__vy[i]
            v2=self.__vy[i+1]
            A=np.array([[1,t1,t1**2,t1**3],[1,t2,t2**2,t2**3],[0,1,2*t1,3*t1**2],[0,1,2*t2,3*t2**2]])
            B=np.array([x1,x2,v1,v2])
            coef=scipy.linalg.solve(A,B)
            self.__coefy.append(coef)      
            
            x1=self.__z[i]
            x2=self.__z[i+1]
            v1=self.__vz[i]
            v2=self.__vz[i+1]
            A=np.array([[1,t1,t1**2,t1**3],[1,t2,t2**2,t2**3],[0,1,2*t1,3*t1**2],[0,1,2*t2,3*t2**2]])
            B=np.array([x1,x2,v1,v2])
            coef=scipy.linalg.solve(A,B)
            self.__coefz.append(coef)   
        self.__coefx=np.array(self.__coefx)
        self.__coefy=np.array(self.__coefy)
        self.__coefz=np.array(self.__coefz)

               
    def get(self,t):
        if t>self.tmax or t<self.tmin:
            print('the time should be at the range of ',self.tmax,'--',self.tmin)
            return 
        try:
            ti=np.max(np.arange(len(self.__t))[self.__t<t])
        except:
            ti=np.min(np.arange(len(self.__t))[self.__t>t])
            ti=ti-1
        POS=np.array([self.__coefx[ti][0]+self.__coefx[ti][1]*t+self.__coefx[ti][2]*t**2+self.__coefx[ti][3]*t**3,
                      self.__coefy[ti][0]+self.__coefy[ti][1]*t+self.__coefy[ti][2]*t**2+self.__coefy[ti][3]*t**3,
                      self.__coefz[ti][0]+self.__coefz[ti][1]*t+self.__coefz[ti][2]*t**2+self.__coefz[ti][3]*t**3
                      ])

        VEL=np.array([self.__coefx[ti][1]+2*self.__coefx[ti][2]*t+3*self.__coefx[ti][3]*t**2,
                      self.__coefy[ti][1]+2*self.__coefy[ti][2]*t+3*self.__coefy[ti][3]*t**2,
                      self.__coefz[ti][1]+2*self.__coefz[ti][2]*t+3*self.__coefz[ti][3]*t**2
                      ])
        return POS,VEL
    
