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

## lu_factor

## Author: Matt <matt@mjg-virtualbox>
## Created: 2014-08-01

function [L, U] = lu_factor (A)

[n,m] = size(A)

L = eye(n,n);

U(1,1) = A(1,1)

for i=2:n
    U(1,i) = A(1,i)
    L(i,1) = A(i,1)/U(1,1)
end

for i=2:n-1
    for j=i:n
        sum = 0;
        for k=1:i-1
            sum = sum + L(i,k)*U(k,j)
        end
        U(i,j) = A(i,j) - sum
        
        sum = 0;
        for k=1:i-1
            sum = sum + L(j,k)*U(k,i)
        end
        L(j,i) = (A(j,i)-sum)/U(i,i)
    end
end

sum = 0;
for i=1:n-1
    sum = sum + L(n,i)*U(i,n)
end

U(n,n) = A(n,n) - sum;

endfunction
