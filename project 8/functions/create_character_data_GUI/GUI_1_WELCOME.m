function [] = GUI_1_WELCOME()
% initial GUI of the character creation program
    
    % open a window(figure) for the GUI
    % fh: figure handle
    S.fh = figure('units','pixels',...
                'position',[500 300 600 400],...
                'menubar','none',...
                'name','Character Creator',...
                'numbertitle','off',...
                'resize','off',...
                'Color',[0.8588 0.9412 0.8275]);
    % create a button for training a new CNN
    % pb: pushbutton
    % create a button for loading an existing CNN
    S.pb2 = uicontrol('style','push',...
                'units','pix',...
                'position',[100 10 400 40],...
                'fontsize',14,...
                'string','Draw a character',...
                'backgroundColor',[0.3020 0.4471 0.7412],...
                'foregroundcolor','w',...
                'callback',{@pb_call,S});
    message = 'WELCOME TO CHARACTER CREATOR. USE THIS PROGRAM TO POPULATE';
    message = sprintf('%s YOUR TRAINING AND REFERENCE SETS.',message);
    % create a textbox containing the welcome message
    % tb: text box
    S.tb = uicontrol('style','text',...
                'unit','pix',...
                'position',[50 180 500 125],...
                'min',0,'max',2,...
                'fontsize',20,...
                'string',message,...
                'BackgroundColor',[1.0000 0.9490 0.7412],...
                'fontweight', 'bold');
end

function [] = pb_call(varargin)
% button callback function for loading an existing CNN
    
    % get the structure containing all the GUI elements
    S = varargin{3};
    % close the current GUI window
    close(S.fh);
    
    % start drawing a character
    image = draw_tool();    
    
    % open the GUI for saving the image
    GUI_2_SAVE(image);
end
