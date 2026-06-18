@echo off
title Pythoncraft Github Guncelleme Araci
color 0B

echo ===================================================
echo     PYTHONCRAFT GITHUB YUKLEME/GUNCELLEME ARACI
echo ===================================================
echo.
echo Bu arac kodlarindaki degisiklikleri Github'a yukler.
echo (Ornek kodlar, dbler ve diger gizli dosyalar .gitignore sayesinde yuklenmez.)
echo.

set /p commitmsg="Guncelleme notunu yazin (Bos birakirsaniz 'Guncelleme' yazilir): "
if "%commitmsg%"=="" set commitmsg=Guncelleme

echo.
echo [1] Dosyalar hazirlaniyor...
"C:\Program Files\Git\cmd\git.exe" add .

echo [2] Degisiklikler kaydediliyor...
"C:\Program Files\Git\cmd\git.exe" commit -m "%commitmsg%"

echo [3] Github'a yukleniyor...
"C:\Program Files\Git\cmd\git.exe" push origin main

echo.
echo ===================================================
echo Islem tamamlandi! Degisiklikler Github'da yayinda.
echo ===================================================
pause
