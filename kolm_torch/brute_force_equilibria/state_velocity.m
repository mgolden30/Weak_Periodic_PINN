function f = state_velocity( state, N, forcing, nu )
  %{
  PURPOSE:
  Compute the time derivative dw/dt of a vorticity field w.

  INPUT:
  state - a [N*N+1,1] matrix containing vorticity and u0.
  N - see above
  forcing - any forcing, same size as w
  nu - viscosity. 1/40 is the current target
  
  OUTPUT:
  f - a vector which is the same size as state
  %}

  w  = reshape( state(1:N*N), [N,N] ); %should be passed in as vector
  u0 = state(N*N+1);

  k = 0:N-1; k(k>N/2) = k(k>N/2) - N;
  kx = k;
  ky = k';

  %Construct spectral operators to go from vorticity to velocity
  k_sq = kx.^2 + ky.^2;
  to_u =  1i*ky./k_sq;
  to_v = -1i*kx./k_sq;
  to_u(1,1) = 0;
  to_v(1,1) = 0;

  wf = fft2(w);
  u  = real(ifft2( to_u .* wf )) + u0;
  v  = real(ifft2( to_v .* wf ));
  wx = real(ifft2( 1i*kx.* wf ));
  wy = real(ifft2( 1i*ky.* wf ));
  dis= real(ifft2( k_sq .* wf ));

  vel = -u.*wx - v.*wy - nu*dis + forcing;

  %Try applying an overall inverse Laplacian
  k_inv = 1./k_sq;
  k_inv(1,1) = 1;
  vel = real(ifft2( k_inv .* fft2(vel) ));
  

  f = zeros(N*N+1,1);
  f(1:N*N) = reshape( vel, [N*N,1] );
  %leave the last element as zero 
end