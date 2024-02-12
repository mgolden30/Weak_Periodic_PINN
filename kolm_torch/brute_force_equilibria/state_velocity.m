function vel = state_velocity( w, N, forcing, nu )
  %{
  Compute the state space velocity of vorticity
  %}

  w = reshape(w, [N,N]); %should be passed in as vector

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
  u  = real(ifft2( to_u .* wf ));
  v  = real(ifft2( to_v .* wf ));
  wx = real(ifft2( 1i*kx.* wf ));
  wy = real(ifft2( 1i*ky.* wf ));
  dis= real(ifft2( k_sq .* wf ));

  vel = -u.*wx - v.*wy - nu*dis + forcing;

  %{
  Try applying an overall inverse Laplacian
  %}
  
  k_inv = 1./k_sq;
  k_inv(1,1) = 1;

  vel = real(ifft2( k_inv .* fft2(vel) ));

  vel = reshape( vel, [N*N,1]);
end