import numpy as np

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

def to_streamfunction( ws ):
    N  = np.shape(ws)[0]

    #need d to get the usual integer k
    k  = np.fft.fftfreq(N, d=1.0/N)
    k  = 1.0*k
    ky, kx = np.meshgrid(k,k)
    
    wf = np.fft.fft2(ws, axes=[0,1]) #take the Fourier transform

    to_psi = 1.0/(kx**2 + ky**2)
    to_psi[0,0] = 0 #kill zero mode of streamfunction

    #add an extra axis for time
    to_psi = to_psi[..., np.newaxis]

    #take derivatives and find flow velocity
    psi = np.real( np.fft.ifft2( wf*to_psi, axes=[0,1] ) )
    return psi

def implicit_explicit_step(w,dt,nu,forcing):
    # Evolve vorticity forward one timestep
    # Handle advection explicitly and dissipation implicitly
    # since we are in the high dissipation regime
    N  = np.shape(w)[0]

    #need d to get the usual integer k
    k  = np.fft.fftfreq(N, d=1.0/N)
    k  = 1.0*k

    #print(k)

    ky, kx = np.meshgrid(k,k)

    #mask for dealiasing, standard 2/3rds
    mask = (kx**2 + ky**2) > (N/3)**2
    

    k1 = right_hand_side(w, kx, ky, forcing, mask)
    
    #create half-step
    wt = w + dt*k1*0.5
    wt = implicit_viscosity(wt, dt/2, nu, kx, ky)

    k2 = right_hand_side(wt, kx, ky, forcing, mask)
    
    w = w + dt*k2
    w = implicit_viscosity( w, dt, nu, kx, ky )

    #Clean up state
    w = dealias(w, mask)
    w  = np.real(w)
    w = w - np.mean(w)

    return w

