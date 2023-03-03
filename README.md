METEOR
Multivariate Emulation of Time-Emergent fOrced Response


PDRMIP_spatial_emulator is working concept

datadir requires the 'training' subdirectory, containing PDRMIP output - linked here https://drive.google.com/drive/folders/1iLy5y1wlKJ-MLuJa7Y35XabgaU1Zzu8G?usp=share_link

DATADIR must be set as an environment variable in a file .env in the model directory, pointing to the 'training' folder

prpatt contains library with functions to fit parameters to define synthetic PC timeseries to processes PDRMIP output, as well as functions to run convolve synthetic PCA with a user-defined forcing timeseries.
