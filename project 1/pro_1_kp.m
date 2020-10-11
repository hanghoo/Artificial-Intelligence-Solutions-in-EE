%{
| Project #1A - Public Transportation Train Budget Planning              |
|------------------------------------------------------------------------|
| The Gotham Transportation Authority is purchasing new trains for its   |
| elevated train lines. The GTA wants to maximize the total number of    |
| daily passengers while putting a constraint on the weight of the train.|
| Due to regulations, GTA cannot purchase more than one car from a given |
| manufacturer.                                                          |
--------------------------------------------------------------------------

%}
clc;
clear;

% set weight_limit, read input file into variable trains_table:
weight_limit = 800;
input_file = 'Train_info.csv';
trains_table = readtable(input_file);

fprintf('******** READING ITEMS FROM %s ********\n',input_file);
fprintf('WEIGHT LIMIT IS SET TO %d\n',weight_limit);

% define chromosome length and fitness function:
chromosome_length = height(trains_table);

fit_func = @(chromosome) - (chromosome * trains_table.daily_passengers);

% define A, b, Lb, Ub, int_indices:
A = trains_table.weight';
b = weight_limit;

Lb = zeros(1, chromosome_length);
Ub = ones(1, chromosome_length);

int_indices = 1:chromosome_length;


% run ga:
disp('****GA STARTING*****');
options = optimoptions('ga','display','off');
[selection, selection_fitness] = ga(fit_func,chromosome_length,A,b,...
                                    [],[],Lb,Ub,[],int_indices);
fprintf('Best fitness for this run = %d\n', abs(selection_fitness));
disp('****GA Finished****');

%display results:
if selection == zeros(chromosome_length, 1)
    message = sprintf('GA CANNOT FIND VALID SELECTION WITH GIVEN CONSTRAINTS');
    disp(message)
else
    message = sprintf('OPTIMAL SELECTION OF ITEMS: [');
    for i = 1:chromosome_length
        if selection(i) == 1
            message = sprintf('%s \n\t- %s', message, string(trains_table.Manufacturer(i)));    
        end
    end

    fprintf('%s\n ]\n', message);
    fprintf('TOTAL weight OF RAILCARS: %d Tons\n', selection * trains_table.weight);
    fprintf('TOTAL DAILY PASSENGERS: %d\n', selection * trains_table.daily_passengers);
end
disp('*********************************************')
