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

## inverse_power_method

## Author: Matt <matt@mjg-virtualbox>
## Created: 2014-07-30

function [mu, x, err, has_converged] = inverse_power_method (A, x, tol, max_iter)

has_converged = false;

q = x'*A*x/(x'*x)
x = x / norm(x, Inf)
I = eye(size(A));

for i = 1:max_iter
    y = inv(A - q * I)*x
    y_max = max(y);
    y_min = min(y);
    if (abs(y_min) > y_max)
        y_p = y_min
    else
        y_p = y_max
    end
    
    x - y/y_p
    err = norm(x - y/y_p, Inf)
    x = y/y_p
    
    if (err < tol)
        has_converged = true;
        break;
    end
end

mu = 1/y_p + q

endfunction
