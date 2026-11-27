@echo off
taskkill /FI "WINDOWTITLE eq Backend*" /F
echo Done. If the Backend window is still open, close it by hand.
pause
