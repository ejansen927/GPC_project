import numpy as np

def CrossProduct(v1,v2):
    x=v1[1]*v2[2]-v1[2]*v2[1]
    y=-(v1[0]*v2[2]-v1[2]*v2[0])
    z=v1[0]*v2[1]-v1[1]*v2[0]

    return np.array([x,y,z])

def PBC(index,Nx):
    if (index >=Nx):
        index=index-Nx
    elif (index<0):
        index=index+Nx
    else:
        index=index
    return index

#The formula for Chirality is used from: https://journals.aps.org/prb/pdf/10.1103/PhysRevB.105.104428
def Chirality(SpinConf):

    # spinconf is (3, Nx, Ny)
    _, Nx, Ny = SpinConf.shape
    local_chi=np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            #print(SpinConf[:,PBC(i+1,Nx),j],SpinConf[:,i,PBC(j+1,Ny])
            cross_r=CrossProduct(SpinConf[:,PBC(i+1,Nx),j],SpinConf[:,i,PBC(j+1,Ny)])
            cross_l=CrossProduct(SpinConf[:,PBC(i-1,Nx),j],SpinConf[:,i,PBC(j-1,Ny)])
                    
            local_chi[i][j]=np.dot(SpinConf[:,i,j],cross_r) + np.dot(SpinConf[:,i,j],cross_l)
    return local_chi/(Nx*Ny)

#this function finds skyrmion number (assuming the charge is +-1)
def SkyrmionNumber(SpinConf):

    # spinconf is (3, Nx, Ny)
    _, Nx, Ny = SpinConf.shape
    local_chi=np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            cross_r=CrossProduct(SpinConf[:,PBC(i+1,Nx),j],SpinConf[:,i,PBC(j+1,Ny)])
            deno_r=1+np.dot(SpinConf[:,PBC(i+1,Nx),j],SpinConf[:,i,PBC(j+1,Ny)])\
                    + np.dot(SpinConf[:,i,j],SpinConf[:,i,PBC(j+1,Ny)])\
                    + np.dot(SpinConf[:,i,j],SpinConf[:,PBC(i+1,Nx),j])
            #if(deno_r==0): #exclude these points
            if(abs(deno_r)<1e-10): #exclude these points
                #print("Alarm")
                sum_r=0
            else:
                sum_r=np.arctan(np.dot(SpinConf[:,i,j],cross_r)/deno_r)


            cross_l=CrossProduct(SpinConf[:,PBC(i-1,Nx),j],SpinConf[:,i,PBC(j-1,Ny)])
            deno_l= 1 + np.dot(SpinConf[:,PBC(i-1,Nx),j],SpinConf[:,i,PBC(j-1,Ny)])\
                    + np.dot(SpinConf[:,i,j],SpinConf[:,i,PBC(j-1,Ny)])\
                    + np.dot(SpinConf[:,i,j],SpinConf[:,PBC(i-1,Nx),j])
            #if(deno_l==0):
            if(abs(deno_l)<1e-10):
                #print(np.dot(SpinConf[:,PBC(i-1,Nx),j],SpinConf[:,i,PBC(j-1,Nx)]))
                #print()
                #print(i,j,np.dot(SpinConf[:,i,j],cross_l),deno_l)
                #print(np.dot(SpinConf[:,i,j],cross_l)/deno_l)
                sum_l=0
            else:
                sum_l=np.arctan(np.dot(SpinConf[:,i,j],cross_l)/deno_l)
            
            local_chi[i][j]=sum_r + sum_l

    return local_chi/(2*np.pi)


