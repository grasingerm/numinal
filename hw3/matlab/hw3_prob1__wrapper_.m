function hw3_prob1__wrapper (n_x, n_y, L, delta, n_terms)
    x_coords = linspace(L/n_x, L-L/n_x, n_x);
    y_coords = linspace(L/n_y, L-L/n_y, n_y);
    
    [A_n,u] = helmholtz (x_coords,y_coords,delta,n_terms);
    
    disp(A_n);
    write_handle = fopen(strcat('helmholtz_N-',num2str(n_x),'.csv'),'w');
    fprintf(write_handle,'%6s,%6s,%6s,%6s,%10s\n','i','j','x','y','u(x,y)');
    [rows,cols] = size(u);
    for i=1:rows
        for j=1:cols
            fprintf(write_handle, '%6d,%6d,%6.2f,%6.2f,%10.6f\n', ...
            i, j, x_coords(i), y_coords(j), u(i,j));
        end
    end
            
    fclose(write_handle);
    surf(x_coords, y_coords, u);
end