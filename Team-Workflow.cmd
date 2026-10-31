@echo off
if "%~1"=="" (
  python "%~dp0scripts\team_workflow.py" status
) else (
  python "%~dp0scripts\team_workflow.py" %*
)
