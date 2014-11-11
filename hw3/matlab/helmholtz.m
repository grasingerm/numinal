function [A_n,u] = helmholtz (x_coords, y_coords, delta, n_terms)
    [garbage,n_x] = size(x_coords);
    [garbage,n_y] = size(y_coords);
    u = zeros(n_x,n_y);
    A_n = zeros(n_terms, 1);
    
    for i=1:n_terms
        n = 2*i-1;
        A_n(i) = -(16.0 / (n^2 * pi^2 * (2 * n^2 * pi^2 + 1/delta) ));
    end
    
    for k=1:n_x
        for j=1:n_y
            x = x_coords(k);
            y = y_coords(j);
        
            for i=1:n_terms
                n = 2*i-1;
                u(k,j) = u(k,j) + A_n(i) * sin(n*pi*x) * sin(n*pi*y);
            end
        end
    end
end
