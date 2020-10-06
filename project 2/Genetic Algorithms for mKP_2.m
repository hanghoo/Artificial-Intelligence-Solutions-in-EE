%{
| Project #2B -  Class Schedule Optimization                             |
--------------------------------------------------------------------------
Given a list of classes at City College of Gotham, Stanley has to pick 
as many classes as he can as long as it meets his financial requirements.  
For example, he can select from several categories: EE, ELECTIVE,ENGR,LIBERAL.
He can take multiple classes from each category.
For example, classLimits = [2, 2, 2, 1]; means he can take up to 2 EE, 
2 ELECTIVE, 2 ENGR and 1 LIBERAL classes. Budget = 3300 

%}
clc;
clear;

% set budget, and read input file into class_table:
budget = 3300;
items_file = 'Class_info.csv';
class_table = readtable(items_file);
class_table = sortrows(class_table,'ClassType');

% Define limits for each class type:
class_limits = [2, 2, 2, 1]; % number of each classType allowed (i.e., max)

% define chromosome and fitness function:
chromosome_length = height(class_table);

fit_func = @(chromosome)- (chromosome * class_table.Cost);
% defined masks based on class_table:
class_index_map = containers.Map();

for i = 1:height(class_table)
    class_type = class_table.ClassType{i};    
    if isKey(class_index_map, class_type)
        indices = class_index_map(class_type);
        indices = horzcat(indices,i);
        class_index_map(class_type) = indices;
    else
        class_index_map(class_type) = [i];
    end
    
end

noof_class_type = size(class_index_map,1);

masks = zeros(noof_class_type, height(class_table));

keySet = keys(class_index_map);

for i = 1:noof_class_type
    indices = class_index_map(keySet{i});
    for j = 1:length(indices)
        masks(i, indices(j)) = 1;
    end
end


% define A, b, Lb, Ub, and int_indices:
A = vertcat(class_table.Cost', masks);
b = [budget, class_limits];
Lb = zeros(1,chromosome_length);
Ub = ones(1,chromosome_length);
int_indices = 1:chromosome_length;

% run ga:
disp('****GA STARTING*****');
options = optimoptions('ga','display','off');
selection = ga(fit_func,chromosome_length,A,b,...
[],[],Lb,Ub,[],int_indices);
disp('****GA Finished****');

% display results:
message = sprintf('OPTIMAL SELECTION OF ITEMS: [');
for i = 1:chromosome_length
    if selection(i) == 1
        message = sprintf('%s \n\t%s - %s', message, string(class_table.Class(i)), ...
            string(class_table.Class_Name(i)));    
    end
end

fprintf('%s \n]\n', message);
fprintf('TOTAL CREDITS TO TAKE THIS SEMESTER: %d\n', selection * class_table.Credits);
fprintf('TOTAL TUITION FOR THIS SEMESTER: $%d\n', selection * class_table.Cost);
disp('*********************************************')
