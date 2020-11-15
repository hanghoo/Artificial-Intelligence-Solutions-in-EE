In this assignment, you will create a CNN to recognize the characters
drawn by the user.

Before starting the assignment, you will create several sub-directories inside 
ReferenceSet and TrainingSet directories:
    (a) In ReferenceSet directory:
        - create a new directory for each character to 
          be recognized; typically the directory is
          named to represent the character (e.g., a directory 
          called "letter_A" is suitable for storing a 
          reference drawing for the letter "A", etc.); 
          note that there should be only one file 
          per character in ReferenceSet directory;
    (b) In TrainingSet directory:
        - create a new directory for each character to be 
          recognized; note that the names of the directories 
          in TrainingSet and ReferenceSet must match;

Below is an example of how the directory tree should look like for 
a CNN that can recognize the characters 'C','?', and '7':

            referenceSet
            |
            |---letter_C
            |
            |---question_mark
            |
            |---seven

            trainingSet
            |
            |---letter_C
            |
            |---question_mark
            |
            |---seven


After the sub-directories are created, they should be populated by
running Matlab program called create_new_character.m:

create_new_character.m: (this file is not to be modified by the students)
    (a) This program creates images for characters to be used both
        as reference images and CNN training images;
    (b) The user is asked to draw a character using a mouse
        (multiple strokes for drawing an image is allowed);
    (c) If the drawn image will be used as a reference image, user 
        will save it inside ReferenceSet under the appropriate sub-directory;
        (for example, under ReferenceSet/Letter_A directory, there will
        be a single image of the letter 'A' saved as Letter_A_ref.png)
    (d) If the drawn image will be used as a training sample image, user 
        will save it inside TrainingSet under the appropriate sub-directory;
        (for example, under TrainingSet/Letter_A directory may contain many
        images of the letter 'A' saved as Letter_A_1.png, Letter_A_2.png, etc.)

Below is an example of how the directory tree should look like for a CNN that
can recognize the characters 'C','?', and '7' after populating it with images:

            referenceSet
            |
            |---letter_C
            |   |---letter_C_ref.png
            |
            |---question_mark
            |   |---question_mark_ref.png
            |
            |---seven
                |---seven_ref.png

            trainingSet
            |
            |---letter_C
            |   |---letter_C_1.png
            |   |---letter_C_2.png
            |   |---letter_C_3.png
            |
            |---question_mark
            |   |---question_mark_1.png
            |   |---question_mark_2.png
            |   |---question_mark_3.png
            |
            |---seven
                |---seven_1.png
                |---seven_2.png
                |---seven_3.png


After the trainingSet and referenceSet directories are are populated with 
images, the CNN can now be defined, trained and used for predictions
as follows:

pro_8_ocr_x.m: (This is the main program that runs the project. Do not
                modify anything in this file except for the function called
                cnn_train())
    (a) This is the main program that runs CNN for the project;
    (b) This program prompts the user to create a new CNN or load an
        existing one;
    (c) If the user chooses to train a new CNN, it calls cnn_train();
    (d) If the user chooses to use a previously trained CNN, it loads 
        a CNN existing in the same directory saved previously as
        a .mat file;
    (e) It then calls cnn_predict(), which asks the user to draw 
        a character;
    (f) cnn_predict then passes the drawn character to the trained CNN 
        which generates an output displaying the predicted character;
    (g) The user is then prompted if new predictions using the same CNN
        will be needed; if so, the above steps (e) and (f) are repeated;
    (h) The user is finally prompted if the trained CNN is to be saved for
        future use; if user chooses to save this CNN, it will be saved in
        the current directory as a mat file (make sure to include .mat
        extension in the file name);

cnn_train(): (this function is located at the bottom of pro_8_ocr_x.m. It is
              given as a skeleton and will be modified by the students)
    (a) This function defines the architecture of CNN as follows:
        - CONV LAYER (size, number of filters)
        - BATCH NORMALIZATION
        - RELU
        - POOL(size, stride)
        - CONV LAYER (size, number of filters)
        - BATCH NORMALIZATION
        - RELU
        - POOL(size, stride)
        - CONV LAYER (size, number of filters)
        - BATCH NORMALIZATION
        - RELU
        - FULLY CONNECTED ANN
    (b) It then trains the CNN;

