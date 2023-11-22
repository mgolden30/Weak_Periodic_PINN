import numpy as np
import matplotlib.pyplot as plt
import os


N = 128  #points per side
M = 2**14  #timesteps
every = 32 #plot every
dt= 0.02 #timestep
nu= 1.0/40 #viscosity

grid_1d = np.arange(N)/N * 2*np.pi

x,y = np.meshgrid( grid_1d, grid_1d )

forcing = 4*np.sin(4*y)

#initial condition
w = np.sin(3*x) + np.cos(x-2*y+1)

#store the entire trajectory
ws = np.zeros([N,N,M])

def implicit_step(w,dt,nu,forcing):
    # Evolve vorticity forward one timestep
    # Handle advection explicitly and dissipation implicitly
    # since we are in the high dissipation regime
    N  = np.shape(w)[0]

    #need d to get the usual integer k
    k  = np.fft.fftfreq(N, d=1.0/N)
    k  = 1.0*k

    #print(k)

    ky, kx = np.meshgrid(k,k)

    def right_hand_side( w, kx, ky, forcing, mask ):
        w = dealias(w, mask)
        wf = np.fft.fft2(w) #take the Fourier transform
        i = 1j
        to_u =  i*ky/(kx**2 + ky**2)
        to_v = -i*kx/(kx**2 + ky**2)
        to_u[0,0] = 0 #no mean flow
        to_v[0,0] = 0 #no mean flow

        #take derivatives and find flow velocity
        w_x = np.real( np.fft.ifft2( i*kx*wf ) )
        w_y = np.real( np.fft.ifft2( i*ky*wf ) )
        u   = np.real( np.fft.ifft2( to_u*wf ) )
        v   = np.real( np.fft.ifft2( to_v*wf ) )
    
        #explicit right hand side
        rhs = -u*w_x - v*w_y + forcing
        return rhs
    
    #mask for dealiasing
    mask = (kx**2 + ky**2) > (N/8)**2
    def dealias( w, mask ):
        wf = np.fft.fft2(w)
        wf[mask] = 0
        w = np.real( np.fft.ifft2(wf) )
        return w
    
    def implicit_viscosity( w, dt, nu, kx, ky ):
        #Take the implicit step
        wf = np.fft.fft2(w)
        wf = wf * np.exp( -nu*dt*(kx**2 + ky**2) )
        w  = np.fft.ifft2(wf)
        w  = np.real( w ) 
        return w
    
    #Half an implicit step
    w = implicit_viscosity( w, dt/2, nu, kx, ky )
    w = dealias(w, mask)

    #Take the explicit step with 2nd order Runge-Kutta
    k1 = right_hand_side(w, kx, ky, forcing, mask)
    wt = w + dt*k1*0.5
    k2 = right_hand_side(wt, kx, ky, forcing, mask)
    w = w + dt*k2

    #second implicit step
    w = implicit_viscosity( w, dt/2, nu, kx, ky )
    w = dealias(w, mask)

    w  = np.real(w)

    #Set mean voritcity to zero
    w = w - np.mean(w)

    return w


current_directory = os.getcwd()
# Define the output directory using an absolute path
output_directory = os.path.join(current_directory, 'frames/')
for i in range(M):
    print(i)
    ws[:,:,i] = w #save the state
    w = implicit_step(w,dt,nu,forcing)

    if i % every != 0 :
        continue
    plt.clf()
    plt.imshow(w, cmap='bwr')
    plt.colorbar(label='w')
    #plt.clim(-1,1)
    plt.title(f'Time Step {i}')
    plt.xlabel('y')
    plt.ylabel('x')
        
    # Save the figure as an image
    plt.savefig( '%s/frame_%04d.png' % (output_directory, i) )


#plot low dimensional projection of trajectory

e1 = np.mean( np.reshape(ws**2, [N*N,M]), axis=0 )
e2 = np.mean( np.reshape(ws*forcing[:,:,np.newaxis], [N*N,M]), axis=0 )
plt.clf()
plt.scatter( e1, e2 )
plt.title('Projection')
plt.savefig( '%s/state_space.png' % (output_directory) )

# Make a recurrence diagram
d = int(M/every)
dist = np.zeros([d,d])

for i in range(d):
    for j in range(d):
        dist[i,j] = np.linalg.norm( ws[:,:,i*every] - ws[:,:,j*every] )

plt.clf()
plt.imshow(dist)
plt.title('Recurrence')
plt.clim(0, 500)
plt.colorbar(label='dist')
plt.savefig( '%s/recurr.png' % (output_directory) )