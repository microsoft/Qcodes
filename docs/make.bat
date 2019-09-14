@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "clean" goto clean


%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

REM generate api docs
REM This is not part of any job since it should be done
REM for all jobs except clean and help.
REM Note that all folders after the first one are excluded
REM (see sphinx-apidoc help for more info).
REM Also note that exclusion of "keysight" (lower-case name)
REM is due to duplication of the folder in git that happened
REM a long time ago (i.e. "Keysight", the upper-case, is used
REM for storing drivers, not the lower-case one).
sphinx-apidoc  -o  _auto  -d 10 ..\qcodes ^
    ..\qcodes\instrument_drivers\Spectrum\pyspcm.py ^
    ..\qcodes\instrument_drivers\Spectrum\M4i.py ^
    ..\qcodes\instrument_drivers\Keysight\keysightb1500\* ^
    ..\qcodes\instrument_drivers\QuantumDesign\DynaCoolPPMS\private\*
mkdir api\generated\
copy _auto\qcodes.instrument_drivers.* api\generated\

if "%1" == "htmlfast" goto htmlfast
if "%1" == "htmlapi" goto htmlapi

REM default build used if no other brach is used
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:htmlfast
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O% -D nbsphinx_execute=never
goto end

REM leftover for backwards compatibility. Equivalent to html
:htmlapi
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:clean
del /q /s "_auto"
del /q /s "api\generated"
%SPHINXBUILD% -M clean %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:end
popd
