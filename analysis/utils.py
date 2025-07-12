import freud
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import signac


def unwrap2D(job):
    #open GSD file
     with gsd.hoomd.open(job.fn('trajectory.gsd'), 'r') as traj:
        #calculate msd 
        box = traj[0].configuration.box
        N = traj[0].particles.N
        nframes = len(traj)

        A = np.zeros((nframes, N, 3))          #creates an array of zeroes of dimension (Nframes, Nparticles, 3D)
        B = np.zeros((nframes, N, 3), dtype = np.int32)            #creates an array of zeroes of dimension (Nframes, Nparticles, 3D) to store images

        for i in range(nframes):
            A[i] = traj[i].particles.position
            B[i] = traj[i].particles.image


        A = A + (B * box[0])
        A[:, :, 2] = 0

     return nframes, N, box, A 
#works for square boxes TODO: update it, use a box matrix to make it work for any arbitrary box 

def unwrap3D(job):
    #open GSD file
     with gsd.hoomd.open(job.fn('trajectory.gsd'), 'r') as traj:
        #calculate msd 
        box = traj[0].configuration.box
        N = traj[0].particles.N

        A = np.zeros((len(traj), N, 3))          #creates an array of zeroes of dimension (Nframes, Nparticles, 3D)
        B = np.zeros((len(traj), N, 3), dtype = np.int32)            #creates an array of zeroes of dimension (Nframes, Nparticles, 3D) to store images

        for i in range(len(traj)):
            A[i] = traj[i].particles.position
            B[i] = traj[i].particles.image


        A = A + (B * box[0])

     return nframes, N, box, A 
#works for square boxes TODO: update it, use a box matrix to make it work for any arbitrary box 

def MSD_2D(job):
    _ , _, _, Positions = unwrap2D(job)

    #calculate displacements
    displacements = Positions - Positions[0]

    #calculates the msd
    msd = np.mean(np.linalg.norm(displacements, axis = 2)**2, axis=1)

    return msd
    

def cr_disp(job):
    nframes, N, box, Positions = unwrap2D(job)

    #calculate displacements
    displacements = Positions - Positions[0]

    #compute neighbor indices using voronoi tesselation
    voro = freud.locality.Voronoi()
    voro.compute((box, Positions[0]))
    nlist = voro.nlist
    neighbor_indices = []
    i = 0
    for j in range(N):
        templist = []
        while (i < len(nlist.query_point_indices)) and (nlist.query_point_indices[i] == j):
            templist.append(nlist.point_indices[i])
            i+=1
        neighbor_indices.append((templist))
    
    # initialize cage_disp as zeros
    cage_disp = np.zeros_like(displacements)  # same shape (Nframes, N, 3)
    
    # loop over particles to find average displacements of their respective cages
    for j in range(N):
        neighbors = neighbor_indices[j]
        if len(neighbors) > 0:
            # get displacements of neighbors: shape (Nframes, len(neighbors), 3)
            neighbor_disp = displacements[:, neighbors, :]
            # take mean along neighbors axis
            mean_neighbor_disp = np.mean(neighbor_disp, axis=1)  # shape (Nframes, 3)
            # store
            cage_disp[:, j, :] = mean_neighbor_disp
        else:
            # if no neighbors, keep as zeros or you could set to np.nan
            pass


    crdisp = (displacements - cage_disp)

    return nframes, N, box, crdisp

def cr_msd(job):
    _, _, _, crdisp = cr_disp(job)
    crmsd = np.mean(np.linalg.norm(crdisp, axis = 2)**2, axis=1)

    return crmsd

def cr_sisf(job, length):
    nframes, N, box, crdisp = cr_disp(job)

    disp2D = crdisp[:,:,:2]

    k_mag = 2 * np.pi / length

    # Generate angles evenly spaced from 0 to 2pi
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)


    k_vectors = np.array([
        [k_mag * np.cos(theta), k_mag * np.sin(theta)]
        for theta in angles
    ])

    sisf = np.zeros(nframes)
    M = len(k_vectors)

    for i in range(nframes):
        tot = 0
        for j in range(N):
            for k in range(M):
                tot +=  np.cos(np.dot(k_vectors[k], disp2D[i][j]))
        sisf[i] = (tot / (N * M))

    return sisf
    