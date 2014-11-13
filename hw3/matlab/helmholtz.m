function [A_nm,u] = helmholtz (x_coords, y_coords, delta, n_terms)
    [garbage,n_x] = size(x_coords);
    [garbage,n_y] = size(y_coords);
    u = zeros(n_x,n_y);
    A_nm = zeros(n_terms, n_terms);
    
    for i=1:n_terms
        for j=1:n_terms
            n = 2*i-1;
            m = 2*j-1;
            A_nm(i,j) = (16.0 / ...
                ( (-n^2*pi^2 - m^2*pi^2 - 1/delta) * n*m*pi^2 ) ...
                );
        end
    end
    
    for k=1:n_x
        for p=1:n_y
            x = x_coords(k);
            y = y_coords(p);
        
            for i=1:n_terms
                for j=1:n_terms
                    n = 2*i-1;
                    m = 2*j-1;
                    u(k,p) = u(k,p) + A_nm(i,j) * sin(n*pi*x) * sin(m*pi*y);
                end
            end
        end
    end
end
