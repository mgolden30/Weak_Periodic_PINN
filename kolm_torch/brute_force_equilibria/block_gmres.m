function [x, residual] = block_gmres(A, b, X0, tol, inner, outer)
  %{
  PURPOSE:
  This function solves the linear system Ax=b through power iteration of A.
  Instead of performing power iteration on a single vector as is done in GMRES,
  a block of vectors is iterated. This provides more control over the
  Krylov subspace and allows parallel matrix evaluations if you want to program that.

  INPUT:
  A - a function handle to the matrix
  b - the right hand side. size = [n,1]
  X0 - block to perform power iteration on. size = [n,m], where m is the block size
       Note: b will be appended to X0 automatically, so X0 = [] is
       equivalent to ordinary GMRES.
  tol - a tolerance used in solving the reduced matrix. This is the
        smallest number you are willing to divide by at trust the result.

  OUTPUT:
  x - the solution
  residual - norm(A*x-b)/norm(b)
  %}

  n = size(b,  1);
  m = size(X0, 2) + 1;

  %approximately Hessenberg matrix
  H = zeros( m*(inner+1), m*inner );
  Q = zeros( n, m*inner); %orthonormal basis from power iteration
  x = zeros( n, 1); %solution vector we will build iteratively

  %normalize the right hand side for numerical conditioning.
  norm_b = norm(b);
  b = b/norm_b;

  %Loop over outer iterations in which Arnoldi iteration is restarted
  for j = 1:outer
  
    %Modify X0 to contain the right hand side
    X0_mod = [X0, b];
    [Q(:,1:m), ~] = qr( X0_mod, "econ" );
    %^find orthonormal basis of X0 with QR decomposition
    
    %Keep track of how many subsapce dimensions we actually use
    subspace_dim = 0;

    for i = 1:inner
      %Do a loop over inner iterations
      current_block = (i-1)*m + (1:m);
      next_block    = (i  )*m + (1:m);
      past_blocks   = 1:i*m;

      %Evaluate on the current block
      Aq = A(Q( :, current_block )); %replace with matrix multiplication
      H( past_blocks, current_block ) = Q( :, past_blocks )' * Aq; %Project onto past basis
      Aq = Aq - Q(:,past_blocks)*H(past_blocks, current_block); %Orthogonalize with respect to this basis
      
      %Look at what is left over after projecting.
      %Be ready to through out vectors we do not need.
      [U, R] = qr( Aq, "econ" );

      Q(:,next_block) = U;
      H(next_block, current_block) = R;
      
    
    end
  
    %Project b into the Krylov subspace
    b2 = Q'*b;
    x2 = careful_QR_solve( H, b2, tol );
    %x2 = H\b2;

    Ax2 = Q*H*x2;
    x2  = Q( :, 1:inner*m )*x2; %Project back into physical space
    
    %update x
    x = x + x2;
    %update b
    b = b - Ax2;

    %If you want to check how orthonormal things were
    %{
    figure(2);
    imagesc(Q'*Q);
    drawnow
    %}
  end

  %estimate the residual after all of the outer iterations.
  residual = norm( b );
  %rescale x by the original norm of b
  x = x*norm_b;
end


function x = careful_QR_solve(A, b, tol)
  %compute qr with pivoting
  [Q,R,P] = qr( A, "vector" );
  
  %check that I understand QR decompositions
  %max(max( A(:,P) - Q*R ))
  
  n = size(A,2);
  
  R = R(1:n, 1:n);
  Q = Q(:, 1:n);
  b = Q'*b;

  x = 0 *b;
  for i = n:-1:1
    if( abs(R(i,i)) < tol )
      %We don't trust this diagonal element to maintain 
      %numerical stability
      continue;
    else
      x(i) = (b(i) - R(i,(i+1):n)*x((i+1):n))/R(i,i);
    end
  end
  %apply the permutation
  x(P) = x;
end