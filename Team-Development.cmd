@echo off
if "%~1"=="" (
  python "%~dp0scripts\team_development.py" --state "%~dp0..\team-development-20261029" status
) else (
  python "%~dp0scripts\team_development.py" --state "%~dp0..\team-development-20261029" %*
)
