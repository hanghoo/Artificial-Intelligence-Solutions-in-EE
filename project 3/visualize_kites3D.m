function visualize_kites3D(kite_positions, kite_moves, num_kites, rad_dist)
    figure;
   
    [unit_sph_x, unit_sph_y, unit_sph_z] = sphere; 
    
    min_x = min(min(kite_positions(:,1,:))) - rad_dist;
    max_x = max(max(kite_positions(:,1,:))) + rad_dist;
    min_y = min(min(kite_positions(:,2,:))) - rad_dist;
    max_y = max(max(kite_positions(:,2,:))) + rad_dist;
    min_z = 0;
    max_z = max(max(kite_positions(:,3,:))) + rad_dist;
    x_span = (max_x - min_x);
    y_span = (max_y - min_y);
    x_axis_min = (round(min_x/rad_dist)-1)*rad_dist;
    x_axis_max = (round(max_x/rad_dist)+1)*rad_dist;
    y_axis_min = (round(min_y/rad_dist)-1)*rad_dist;
    y_axis_max = (round(max_y/rad_dist)+1)*rad_dist;
    z_axis_min = min_z;
    z_axis_max = (round(max_z/rad_dist)+1)*rad_dist;
    iconsize = [x_span/30 y_span/30];
    
    for t = 1:kite_moves
       pause(0.25); 
       hold off
       
       for kite_num = 1:num_kites
            
            x_pos = kite_positions(kite_num,1,t);
            y_pos = kite_positions(kite_num,2,t);
            z_pos = kite_positions(kite_num,3,t);

            sph_x = (rad_dist*unit_sph_x + x_pos);
            sph_y = (rad_dist*unit_sph_y + y_pos);
            sph_z = (rad_dist*unit_sph_z + z_pos);
            surf(sph_x, sph_y, sph_z,'FaceColor', 'blue', ...
                 'LineStyle', '-', 'EdgeColor', 'blue', ... 
                 'FaceAlpha', 0.05, 'EdgeAlpha', 0.1);
            xlim([x_axis_min x_axis_max])
            ylim([y_axis_min y_axis_max])
            zlim([z_axis_min z_axis_max])
            hold on
            
            scatter3(x_pos, y_pos, z_pos, 'filled');
       end
       patch([x_axis_max x_axis_min x_axis_min x_axis_max], ...
                    [y_axis_max y_axis_max y_axis_min y_axis_min], [0 0 0 0], ...
                    [3 135 36]./255, 'FaceAlpha', 0.5, 'EdgeColor', [0 0 0], ...
                    'EdgeAlpha', 1);
       plot_title = sprintf('Kite Positions at t = %d', t);
       title(plot_title);
    end
end