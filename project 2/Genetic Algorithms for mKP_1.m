%{
| Project #2A - Public Transportation Bus Budget Planning                |

The Gotham Transportation Authority is purchasing new buses to add to 
their current bus network. To honor a previous agreement with bus 
manufactorers, the GTA will buy one bus from each manufacturer. The GTA
would like to maximize the total number of passengers in the new buses
while staying below a given budget and fuel comsumption limit. 
%}
clc;
clear;

% set up money_to_spend and gasoline_consumption:
money_to_spend = 3000;
gasoline_consumption = 80;

% read input file into bus_table:
input_file = 'Bus_info.csv';
bus_table = readtable(input_file);
fprintf('********** READING ITEMS FROM %s ************\n', input_file);
fprintf('MONEY TO SPEND IS SET TO %d\n', money_to_spend);
fprintf('GASOLINE CONSUMPTION IS SET TO %d\n', gasoline_consumption);

% define chromosome length and fitness function:
chromosome_length = height(bus_table);

fit_func = @(chromosome)-(chromosome * bus_table.Passengers);

% define masks based on bus_table:
% generate an empty map containing the indices for each unique category
category_index_map = containers.Map();

% iterate through all the companies in bus_table
for i = 1:height(bus_table)
    % get the category of the companies
    category = bus_table.Companies{i};
    
    if isKey(category_index_map, category)
        indices = category_index_map(category);
        indices = horzcat(indices, i);
        category_index_map(category) = indices;
    else
        % add a new key and add a list containing only the current index
        category_index_map(category) = [i];
    end
    
end

% generate the masks
% get the number of unique categories, e.g., 4
noof_categories = size(category_index_map,1);

masks = zeros(noof_categories, height(bus_table));

keySet = keys(category_index_map);

for i = 1:noof_categories
    indices = category_index_map(keySet{i});
    for j = 1:length(indices)
        masks(i,indices(j)) = 1;
    end    
end

%set A, b, Lb, Ub, int_indices:
A = vertcat(bus_table.Costs', bus_table.Gasoline',masks);
b = [money_to_spend, gasoline_consumption, ones(1,noof_categories)];

Lb = zeros(1,chromosome_length);
Ub = ones(1, chromosome_length);
int_indices = 1:chromosome_length;

% run ga:
disp('****GA STARTING*****');
options = optimoptions('ga','display','off');
selection = ga(fit_func,chromosome_length,A,b,...
[],[],Lb,Ub,[],int_indices);
disp('****GA Finished****');

if isempty(selection)
    message = sprintf('GA CANNOT FIND VALID SELECTION WITH GIVEN CONSTRAINTS');
    disp(message)
    return
end

% display results:
message = sprintf('OPTIMAL SELECTION OF ITEMS: [');
for i = 1:chromosome_length
    if selection(i) == 1
            message = sprintf('%s \n\t %s - %s', message, ... 
                string(bus_table.Companies(i)), string(bus_table.Type(i)));
    end
end
fprintf('%s \n]\n', message);
fprintf('TOTAL PASSENGERS: %d\n', selection * bus_table.Passengers);
fprintf('TOTAL MONEY SPENT: $%dM\n', selection * bus_table.Costs);
fprintf('TOTAL GAS USED: %dK GALLONS\n', selection * bus_table.Gasoline);
disp('*********************************************')

