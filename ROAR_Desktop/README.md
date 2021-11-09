# ROAR_GUI

to rebuild ui, enter the command 
`pyuic5 -x UI_FILE_NAME.ui > [YOUR PROJECT DIRECTORY]/ROAR_GUI/view/UI_FILE_NAME.ui.py

You might need to change the generated file's class name to desired name manually.

-x flag will automatically generate a if __name__ == "__main__" runner section which maybe helpful for debugging