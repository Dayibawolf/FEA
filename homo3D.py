import os
import csv
import numpy as np
from numpy.matlib import repmat, ravel
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, spsolve_triangular, cg
    

def homo3D(E, mu, voxel, size = 1):

########################################################################
# size     = element size
# lambda   = Lame's first parameter for solid materials.
# mu       = Lame's second parameter for solid materials.
# voxel    = Material indicator matrix. Used to determine nelx/nely/nelz
########################################################################
    ## INITIALIZE
    nelx,nely,nelz = voxel.shape
    nel = nelx*nely*nelz
    lambda_ = mu*E/((1+mu)*(1-2*mu))
    mu = E/(2*(1+mu))
    # Stiffness matrix
    keLambda,keMu,feLambda,feMu=hexahedron(size/2, size/2, size/2)
   
    # Node numbers and element degrees of freedom for full (not periodic) mesh
    nodenrs=np.reshape(range(1,(1+nelx)*(1+nely)*(1+nelz)+1),(1+nelx,1+nely,1+nelz), order = 'F')
    edofVec=np.reshape(3*nodenrs[:-1,:-1,:-1]+1,(nel,1),order = 'F')
    addx=np.array([0, 1, 2,-3,-2,-1, 3*nelx, 3*nelx+1, 3*nelx+2, 3*nelx+3, 3*nelx+4, 3*nelx+5])
    addxy=3*(nely + 1)*(nelx + 1) + addx
    edof=repmat(edofVec,1,24) + repmat(np.hstack((addx,addxy)),nel,1)
   
    ## IMPOOSE PERIODIC BOUNDARY CONDITIONS
    # Use original edofMat to index into list with the periodic dofs
    nn=(nelx + 1)*(nely + 1)*(nelz + 1)
    nnP=nelx*nely*nelz
    nnPArray = np.zeros((nelx+1,nely+1,nelz+1),dtype = int)
    nnPArray[:-1,:-1,:-1]=np.reshape(range(1,nnP+1),(nelx,nely,nelz), order = 'F')
    # Extend with a mirror of the back border
    nnPArray[-1,:,:] = nnPArray[0,:,:]
    nnPArray[:,-1,:] = nnPArray[:,0,:]
    nnPArray[:,:,-1] = nnPArray[:,:,0]
    
    # Make a vector into which we can index using edofMat:
    dofVector=np.zeros(3*nn, dtype = int)
    dofVector[0:3*nn:3]=3*nnPArray.reshape(-1,order = 'F') - 2
    dofVector[1:3*nn:3]=3*nnPArray.reshape(-1,order = 'F') - 1
    dofVector[2:3*nn:3]=3*nnPArray.reshape(-1,order = 'F')
    edof=dofVector[edof-1]
    ndof=3*nnP
    
    ## ASSEMBLE GLOBAL STIFFNESS MATRIX AND LOAD VECTORS
    # Indexing vectors
    iK=np.kron(edof-1,np.ones((24,1),dtype=int)).T
    jK=np.kron(edof-1,np.ones((1,24),dtype=int)).T
    
    # Material properties assigned to voxels with materials
    lambda_=lambda_*(voxel == 0)
    mu=mu*(voxel == 0)
    
    # The corresponding stiffness matrix entries
    sK=keLambda.reshape(-1,1,order='F').dot(lambda_.reshape(-1,1,order='F').T) + keMu.reshape(-1,1,order='F').dot(mu.reshape(-1,1,order='F').T)
    K=coo_matrix((ravel(sK,order='F'),(ravel(iK,order='F'),ravel(jK,order='F'))),shape=(ndof,ndof)).tocsc()
    K=1/2*(K + K.T)
    
    # Assembly three load cases corresponding to the three strain cases
    iF=repmat((edof-1).T,6,1)
    jF=np.vstack((np.ones((24,nel))-1, 2*np.ones((24,nel))-1, 3*np.ones((24,nel))-1, 4*np.ones((24,nel))-1, 5*np.ones((24,nel))-1, 6*np.ones((24,nel))-1))
    sF=feLambda.reshape(-1,1,order='F').dot(lambda_.reshape(-1,1,order='F').T) + feMu.reshape(-1,1,order='F').dot(mu.reshape(-1,1,order='F').T)
    F = coo_matrix((ravel(sF,order='F'),(ravel(iF,order='F'),ravel(jF,order='F'))),shape=(ndof,6)).tocsc()
    
    ## SOLUTION
    activedofs=edof[ravel(voxel == 0,order='F'),:]
    activedofs=np.sort(np.unique(ravel(activedofs)))
    X=np.zeros((ndof,6))

    #solve by CG method, remember to constrain one node
    KK = K[activedofs[3:]-1,:][:,activedofs[3:]-1]
    for i in range(6):
        X[activedofs[3:]-1,i], info = cg(KK,F[activedofs[3:]-1,i].toarray())
        print(info, end = '>', flush=True)
   
    ## HOMOGENIZATION
    # The displacement vectors corresponding to the unit strain cases
    X0=np.zeros((nel,24,6))
    # The element displacements for the six unit strains
    X0_e=np.zeros((24,6))
    ke=keMu + keLambda
    fe=feMu + feLambda
    #fix degrees of nodes [1 2 3 5 6 12];
    free=np.setdiff1d(range(24),[0,1,2,4,5,11])
    X0_e[free,:]=np.linalg.solve(ke[free,:][:,free],fe[free,:])
    X0[:,:,0]=np.kron(X0_e[:,0].T,np.ones((nel,1)))
    X0[:,:,1]=np.kron(X0_e[:,1].T,np.ones((nel,1)))
    X0[:,:,2]=np.kron(X0_e[:,2].T,np.ones((nel,1)))
    X0[:,:,3]=np.kron(X0_e[:,3].T,np.ones((nel,1)))
    X0[:,:,4]=np.kron(X0_e[:,4].T,np.ones((nel,1)))
    X0[:,:,5]=np.kron(X0_e[:,5].T,np.ones((nel,1)))
    CH=np.zeros((6,6))
    volume=nelx*nely*nelz*size**3
    for i in range(6):
        for j in range(6):
            sum_L = np.multiply((X0[:,:,i]-X[:,i][edof-1]).dot(keLambda), X0[:,:,j]-X[:,j][edof-1])
            sum_M = np.multiply((X0[:,:,i]-X[:,i][edof-1]).dot(keMu)    , X0[:,:,j]-X[:,j][edof-1])
            sum_L = np.reshape(np.sum(sum_L,1),(nelx,nely,nelz),order = 'F')
            sum_M = np.reshape(np.sum(sum_M,1),(nelx,nely,nelz),order = 'F')
            CH[i,j]=1/volume*np.sum(np.sum(np.sum(np.multiply(lambda_,sum_L) + np.multiply(mu,sum_M))))
    CH = np.where(np.abs(CH)<1e-10,0,CH)
    return CH

def hexahedron(a=None,b=None,c=None):
    # Constitutive matrix contributions
    CMu=np.diag([2,2,2,1,1,1])
    CLambda=np.zeros((6,6))
    CLambda[0:3,0:3]=1
    # Three Gauss points in both directions
    xx=np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    yy=np.copy(xx)
    zz=np.copy(xx)
    ww=np.array([5/9, 8/9, 5/9])
    # Initialize
    keLambda=np.zeros((24,24))
    keMu=np.zeros((24,24))
    feLambda=np.zeros((24,6))
    feMu=np.zeros((24,6))
    for ii in range(len(xx)):
        for jj in range(len(yy)):
            for kk in range(len(zz)):
                #integration point
                x=xx[ii]
                y=yy[jj]
                z=zz[kk]
                qx=np.array([-((y-1)*(z-1))/8, ((y-1)*(z-1))/8, -((y+1)*(z-1))/8,((y+1)*(z-1))/8, ((y-1)*(z+1))/8, -((y-1)*(z+1))/8,((y+1)*(z+1))/8, -((y+1)*(z+1))/8])
                qy=np.array([-((x-1)*(z-1))/8, ((x+1)*(z-1))/8, -((x+1)*(z-1))/8,((x-1)*(z-1))/8, ((x-1)*(z+1))/8, -((x+1)*(z+1))/8,((x+1)*(z+1))/8, -((x-1)*(z+1))/8])
                qz=np.array([-((x-1)*(y-1))/8, ((x+1)*(y-1))/8, -((x+1)*(y+1))/8,((x-1)*(y+1))/8, ((x-1)*(y-1))/8, -((x+1)*(y-1))/8,((x+1)*(y+1))/8, -((x-1)*(y+1))/8])
                J =np.vstack((qx,qy,qz)).dot(np.vstack(([- a,a,a,- a,- a,a,a,- a],[- b,- b,b,b,- b,- b,b,b],[- c,- c,- c,- c,c,c,c,c])).T)
                qxyz=np.linalg.solve(J,np.vstack((qx,qy,qz)))
                B_e=np.zeros((6,3,8))
                for i_B in range(8):
                    B_e[:,:,i_B]=np.vstack(([qxyz[0,i_B], 0          , 0          ],
                                            [0          , qxyz[1,i_B], 0          ],
                                            [0          , 0          , qxyz[2,i_B]],
                                            [qxyz[1,i_B], qxyz[0,i_B], 0          ],
                                            [0          , qxyz[2,i_B], qxyz[1,i_B]],
                                            [qxyz[2,i_B], 0          , qxyz[0,i_B]]))
                B=np.hstack((B_e[:,:,0],B_e[:,:,1],B_e[:,:,2],B_e[:,:,3],B_e[:,:,4],B_e[:,:,5],B_e[:,:,6],B_e[:,:,7]))
                #Weight factor at this point
                weight=np.linalg.det(J)*ww[ii] * ww[jj] * ww[kk]
                #Element matrices
                keLambda= keLambda + weight*B.T.dot(CLambda).dot(B)
                keMu    = keMu     + weight*B.T.dot(CMu).dot(B)
                #Element loads
                feLambda= feLambda + weight*B.T.dot(CLambda)
                feMu    = feMu     + weight*B.T.dot(CMu)
    return keLambda,keMu,feLambda,feMu

def fileInCurrentDir(text, path=''):
    if not path: path = os.getcwd()
    fileName = os.listdir(path)
    file = []
    for f in fileName:
        if(os.path.splitext(f)[1]==text):
            file.append(f)
    return file
    
if __name__ == '__main__':
    npyPath='D:/Work/W211018-Microstructure/Micro-FEM/version-1.0/npy_file'
    file = fileInCurrentDir('.npy',path=npyPath)
    for i, name in enumerate(file):
        print(('\rNo.{},Total:{}:|            |'+'\b'*13).format(i+1,len(file)), end='',flush=True)
        voxel = np.load(npyPath+'/'+name)
        CH = homo3D(1,0.3,voxel=voxel)
        with open('results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name,])
            writer.writerows(CH)
    #print(CH)
    #os.system('pause')
    
