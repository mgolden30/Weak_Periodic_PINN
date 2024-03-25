function Je = velocity_jacobian( state, e, N, forcing, nu )
  %{
  Compute the exact action of the Jacobian on e
  %}

  %Code this to evaluate the action of the Jacobian on many tangent vectors
  %at the same time. This allows block_GMRES to work.
  m = size(e,2);

  %Extract fields
  w  = reshape( state(1:N*N), [N,N]   );
  u0 = reshape( state(N*N+1), [1,1]   );
  ew = reshape( e(1:N*N,:),   [N,N,m] );
  eu0= reshape( e(N*N+1,:),   [1,1,m] );

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
  ewf= fft2(ew);

  u   = real(ifft2( to_u .* wf )) + u0;
  v   = real(ifft2( to_v .* wf ));
  wx  = real(ifft2( 1i*kx.* wf ));
  wy  = real(ifft2( 1i*ky.* wf ));
  
  eu  = real(ifft2( to_u .* ewf )) + eu0;
  ev  = real(ifft2( to_v .* ewf ));
  ewx = real(ifft2( 1i*kx.* ewf ));
  ewy = real(ifft2( 1i*ky.* ewf ));
  edis= real(ifft2( k_sq .* ewf )); %laplacian of w

  %Take the derivative
  %f = -u.*wx - v.*wy - nu*dis + forcing;
  df  = -u.*ewx - v.*ewy ...
        -eu.*wx - ev.*wy ...
        -nu*edis;
  
  %Try taking the anti-Laplacian
  k_inv = 1./k_sq;
  k_inv(1,1) = 1;
  df = real(ifft2( k_inv.*fft2(df) ));
  
  Je = zeros([N*N+1,m]);
  Je(1:N*N, :) = reshape( df, [N*N,m] );
  Je(N*N+1, :) = sum(wx.*ew, [1,2]); %dot product with x derivative.
end