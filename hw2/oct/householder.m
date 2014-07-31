## Copyright (C) 2014 Matt
## 
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with Octave; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

## householder

## Author: Matt <matt@mjg-virtualbox>
## Created: 2014-07-31

function [A] = householder (A)

[n,m] = size(A);

for k=1:n-2
    q = 0;
    for j=k+1:n
        q = q + (A(j,k))^2;
    end
    
    if (A(k+1,k) == 0)
        alpha = -sqrt(q);
    else
        alpha = -sqrt(q)*A(k+1,k) / abs(A(k+1,k));
    end
    
    rsq = alpha^2 - alpha*A(k+1,k);
    v(k) = 0;
    v(k+1) = A(k+1,k) - alpha;
    
    for j=k+2:n
        v(j) = A(j,k);
    end
    
    for j=k:n
        sum = 0;
        for i=k+1:n
            sum = sum + A(j,i)*v(i);
        end
        u(j) = 1/rsq * sum;
    end
    
    v_dot_u = 0;
    for i=k+1:n
        v_dot_u = v_dot_u + v(i)*u(i);
    end
    
    for j=k:n
        z(j) = u(j) - v_dot_u / (2*rsq) * v(j);
    end
    
    for m=k+1:n-1
        for j=m+1:n
            A(j,m) = A(j,m) - v(m)*z(j) - v(j)*z(m);
            A(m,j) = A(j,m);
            A(m,m) = A(m,m) - 2*v(m)*z(m);
        end
    end
    
    A(n,n) = A(n,n) - 2*v(n)*z(n);
    
    for j=k+2:n
        A(k,j) = 0;
        A(j,k) = 0;
    end
    
    A(k+1,k) = A(k+1,k) - v(k+1) * z(k);
    A(k,k+1) = A(k+1,k);
end

endfunction
