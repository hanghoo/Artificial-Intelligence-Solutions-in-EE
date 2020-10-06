%% Kite Flying with Genetic Algorithm 
% A group of children are flying kites in Seaview Park. A chip is installed
% on the kites to keep a distance between the kites to prevent them from
% colliding with each other. 
clc 
clear
%% User inputs and initalizations
num_kites = 10;
speed = 1; 
rad_dist = 10; % radius
kite_moves = 20 % number of moves 
kite_positions = zeros(num_kites, 3, kite_moves); 

% assign the kites to random locations (ensure that the z is at least
% radius above 0 to avoid collision with ground):
kite_positions(:,3,1) = kite_positions(:,3,1) + rad_dist;

% define chromosome length:
chromosome_length = 5;
    
% define lower bounds and upper bounds of the chromosomes:
Lb = zeros(1,chromosome_length);
Ub = ones(1,chromosome_length);
int_indices = 1:chromosome_length;

% declare the displacement vectors:
displacement_vectors = [[0,1];[1/sqrt(2), 1/sqrt(2)];...
                        [1,0];[1/sqrt(2), -1/sqrt(2)];...
                        [0,-1];[-1/sqrt(2),-1/sqrt(2)];
                        [-1,0];[-1/sqrt(2),1/sqrt(2)]];

displacement_vectors = speed .* displacement_vectors;
 
options = optimoptions('ga','display','off');  
for t = 1:kite_moves-1
    fprintf("\n\n***** TIME IS NOW: %d*****\n", t)
    
    for kite_num = 1:num_kites
        fprintf("***** BEGINNING GA FOR KITE #%d *****\n", kite_num)
        
        % get all neighbors and the position of the current kite:
        neighbors = kite_positions(:,:,t);
        neighbors(kite_num,:) = [];
        current_position = kite_positions(kite_num,:,t); 

        % the fitness function - complete the fitness function definition
        % at the end of the script (second to last function):
        fit_func = @(chromosome) fitness_function(chromosome, ...
                   current_position, speed, neighbors, rad_dist, displacement_vectors);

        selection = ga(fit_func, chromosome_length, [], [], ...
                        [], [], Lb, Ub, [], int_indices, options);

        % invoke matlab genetic algorithm for the current kite:
        displacement = displacement_vectors(bi2de(selection(3:end), 'left-msb')+1, :);
        next_position = get_next_position(selection, current_position, speed, displacement_vectors);

        % get the next position and assign it to the next time step:
        kite_positions(kite_num,:,t+1) = next_position;

        fprintf("***** FINISHED GA FOR KITE #%d AT TIME %d *****\n", ...
            kite_num, t)
    end
end

%% Visualize the Kites
visualize_kites3D(kite_positions, kite_moves, num_kites, rad_dist);
open_anim = true;
while open_anim == true
    fprintf("Would you like to replay the animation?\n\t(1) yes\n\t(2) no\n");
    user_choice = input('');
    if user_choice == 2
        open_anim = false;
        fprintf('Thank you. Have a nice day!\n');
        break;
    else
        close all
        visualize_kites3D(kite_positions, kite_moves, num_kites, rad_dist);
    end
end

function fitness_score = fitness_function(chromosome, position, speed, ...
                            neighbors, rad_dist, displacement_vectors)
    % Write the fitness function:
    fitness_score = 0;
    neighbor_count = 0;
    candidate_pos = get_next_position(chromosome, position, speed, displacement_vectors);
    for i = 1:length(neighbors)
        distance = norm(candidate_pos - neighbors(i,:));
        if distance <= rad_dist
            neighbor_count = neighbor_count + 1;
            fitness_score = fitness_score + (rad_dist - distance);
        end
    end

    if neighbor_count < 1 || candidate_pos(3) <= rad_dist
        fitness_score = abs(intmax);
    end
end

function next_pos = get_next_position(chromosome, position, speed, ...
                                        displacement_vectors)
    if bi2de(chromosome(1:2)) == 0 
        
        next_pos = position;
    elseif bi2de(chromosome(1:2), 'left-msb') == 1 
    
        displacement = displacement_vectors(bi2de(chromosome(3:end), 'left-msb')+1,:);
        next_pos(1:2)= position(1:2) + displacement;
        next_pos(3) = position(3) + speed; 
    elseif bi2de(chromosome(1:2), 'left-msb') == 2
        
        displacement = displacement_vectors(bi2de(chromosome(3:end), 'left-msb')+1,:);
        next_pos(1:2)= position(1:2) + displacement;
        next_pos(3) = position(3) - speed; 
    elseif bi2de(chromosome(1:2), 'left-msb') == 3 
        
        next_pos = position;
        displacement = displacement_vectors(bi2de(chromosome(3:end), 'left-msb')+1,:);
        next_pos(1:2)= position(1:2) + displacement;
    else
    end
end
