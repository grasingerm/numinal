## Copyright (C) 2014 clementine
## 
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
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

## wielandt_deflation

## Author: clementine <clementine@clementine-VirtualBox>
## Created: 2014-07-30

function [mu, x, err, has_converged] = wielandt_deflation (A, x, lambda, tol, max_iter)

[rows, cols] = size(A);

i = 1;
max = 0;
for j = i:rows
    if (abs(x(j)) > max)
        i = j;
    end
end

if (i != 1)
    for k = 1:i-1
        for j = 1:i-1
            B(k,j) = A(k,j) - x(k)/x(i) * A(i,j)
        end
    end
end

if (i != 1 and i != rows)
    for k = i:rows-1
        for j = 1:rows-1
            B(k,j) = A(k+1,j) - x(k+1)/x(i)*A(i,j)
            B(j,k) = A(j,k+1) - x(j)/x(i)*A(i,k+1)
        end
    end
end

if (i != rows)
    for k = i:rows-1
        for j = i:rows-1
            B(k,j) = A(k+1,j+1) - x(k+1)/x(i)*A(i,j+1)
        end
    end
end

endfunction
