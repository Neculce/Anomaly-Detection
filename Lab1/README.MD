So, I have managed to solve the Issue i encountered in the lab, specifically i wasnt able to run my scripts, the interpreted wasnt finding the libraries yet PIP & PIP3 said the libraries were installed. 
I have solved this by moving the script from the root folder of the VENV in the lib/site-packages folder which contains the required libraries. 
Following conclussions:

EX1 : As expected the result of this script is a graph with multiple data points that are classified as outliesr (anomalies) and inliers (regulars)

EX2 : Running the script will show a 2D dataset with a 10% contamination rate along with 2 ROC's computation. One ROC was calculated with 10% contamination rate while the other was calculated with 20%
      - 10% contamination rate yields : TN: 359, FP: 1, FN: 4, TP: 36  and 0.95 Balanced Accuracy
      - 20% contamination rate yields : TN: 331, FP: 29, FN: 1, TP: 39 and 0.95 Balanced Accuracy 
      - The effect on the ROC curve of a higher contamination rate are a higher rate of false positives as the normal points are classified as outliers. We can also see a slight increase in True Positive rate.
      - An observation that I made is that a high contamination rate can lead to a high number of false positives, while a low contamination rate will results in low anomaly detection capabilities and the   
        model might miss some outliers. 

EX3 : The results for EX3 were : dissapointing, I got another error: too many values to unpack. No matter what value n-samples has it always gives this error. This might be due to no testing samples, but I am not sure

EX4 : Running EX4 script returns a more interesting and exciting dataset. We can see multiples outliers that are in the threshold area ( along with the inliers ) which are not detected correctly ( false negatives)
      as well as a couple of inliers which fall outside the threshold and are classified as outliers ( false positives ). As expected all True Outliers outside the threshold are correctly identified. 

Reti M. Matei, SAL - An 1 Master - Anomaly Detection Lab 1
