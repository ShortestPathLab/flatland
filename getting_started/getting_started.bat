@echo on

set FLATLAND_BASEDIR=%~dp0\..

cd %FLATLAND_BASEDIR%

REM uv reads .python-version, provisions the interpreter itself and builds the
REM virtualenv from uv.lock. Install uv first: https://docs.astral.sh/uv/
call uv sync --locked || goto :error
call uv run pytest -v || goto :error
call uv run jupyter notebook || goto :error

goto :EOF


:error
echo Failed with error #%errorlevel%.
pause
