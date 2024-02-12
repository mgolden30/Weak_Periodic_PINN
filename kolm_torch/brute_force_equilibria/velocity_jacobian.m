function Jdw = velocity_jacobian( w, dw, N, forcing, nu )
  %{
  Compute the exact action of the Jacobian on dw
  %}

  w = reshape(w, [N,N]); %should be passed in as vector
  dw= reshape(dw,[N,N]); %should be passed in as vector

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
  dwf= fft2(dw);


  u  = real(ifft2( to_u .* wf ));
  v  = real(ifft2( to_v .* wf ));
  wx = real(ifft2( 1i*kx.* wf ));
  wy = real(ifft2( 1i*ky.* wf ));
  %dis= real(ifft2( k_sq .* wf )); %Don't need

  du  = ifft2( to_u .* dwf );
  dv  = ifft2( to_v .* dwf );
  dwx = ifft2( 1i*kx.* dwf );
  dwy = ifft2( 1i*ky.* dwf );
  ddis= ifft2( k_sq .* dwf );


  %Take the derivative
  %vel = -u.*wx - v.*wy - nu*dis + forcing;
  Jdw = -u.*dwx - v.*dwy ...
        -du.*wx - dv.*wy ...
        -nu*ddis;

  %Try taking the anti-Laplacian
  k_inv = 1./k_sq;
  k_inv(1,1) = 1;
  Jdw = ifft2( k_inv.*fft2(Jdw) );

  Jdw = reshape( Jdw, [N*N,1]);
end