function [A_n,u] = helmholtz (x_coords, y_coords, delta, n_terms)
    [garbage,n_x] = size(x_coords);
    [garbage,n_y] = size(y_coords);
    u = zeros(n_x,n_y);
    A_n = zeros(n_terms, 1);
    
    for i=1:n_terms
        n = i*2;
        A_n(i) = -(64.0 / (n^2 * pi^2 * (2 * n^2 * pi^2 + 1/delta) ));
    end
    
    for i=1:n_x
        for j=1:n_y
            x = x_coords(i);
            y = y_coords(j);
        
            for n=1:n_terms
                u(i,j) = u(i,j) + A_n(n) * sin((2*n-1)*pi*x) * sin((2*n-1)*pi*y);
            end
        end
    end
end
