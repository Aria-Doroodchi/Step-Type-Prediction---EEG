@echo off
echo Starting copy of new files...

robocopy "C:\Users\Aria\Desktop\ML_V2" "C:\Users\Aria\OneDrive - The University of Western Ontario\MSc\Thesis\Data\ML_V2" ^
/E /Z /R:3 /W:5 ^
/COPY:DAT /DCOPY:DAT ^
/XO ^
/Tee /LOG+:C:\logs\new_files_log.txt

echo Copy complete.
pause
