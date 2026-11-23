from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = pytest.mark.win32

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _workflow_script(step_name: str) -> str:
    workflow = (_REPO_ROOT / ".github" / "workflows" / "docs.yaml").read_text(
        encoding="utf-8"
    )
    step = workflow.split(f"    - name: {step_name}\n", 1)[1]
    script = step.split("      run: |\n", 1)[1]
    lines: list[str] = []
    for line in script.splitlines():
        if line and not line.startswith("        "):
            break
        lines.append(line)
    return dedent("\n".join(lines))


@pytest.mark.parametrize("target", ["html", "htmlfast", "htmlapi", ""])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_make_bat_preserves_sphinx_exit_code(
    tmp_path: Path, target: str, exit_code: int
) -> None:
    env = dict(os.environ, SPHINXBUILD=f"cmd /d /c exit {exit_code}")
    result = subprocess.run(
        [
            os.environ["COMSPEC"],
            "/d",
            "/c",
            str(_REPO_ROOT / "docs" / "make.bat"),
            target,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr


def test_make_bat_restores_directory_when_sphinx_is_missing(tmp_path: Path) -> None:
    script = tmp_path / "build.cmd"
    script.write_text(
        f'@echo off\ncall "{_REPO_ROOT / "docs" / "make.bat"}" html\n'
        'set "BUILD_RESULT=%ERRORLEVEL%"\n'
        "cd\nexit /b %BUILD_RESULT%\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [os.environ["COMSPEC"], "/d", "/c", str(script)],
        cwd=tmp_path,
        env=dict(os.environ, SPHINXBUILD="qcodes-nonexistent-sphinx-build"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert str(tmp_path) in result.stdout.splitlines()


@pytest.mark.parametrize(
    "install_codes,available_after,pandoc_exit_code,attempts,success,message",
    [
        ([0], 6, 0, 5, False, "Pandoc executable not found"),
        ([0], 1, 3, 5, False, "pandoc --version failed with exit code 3"),
        ([1], 1, 0, 5, False, "Chocolatey exit code 1"),
        ([1, 0], 2, 0, 2, True, ""),
        ([0], 2, 0, 2, True, ""),
        ([0], 1, 0, 1, True, ""),
        ([1641], 1, 0, 1, True, ""),
        ([3010], 1, 0, 1, True, ""),
    ],
)
def test_pandoc_install_requires_working_executable(
    tmp_path: Path,
    install_codes: list[int],
    available_after: int,
    pandoc_exit_code: int,
    attempts: int,
    success: bool,
    message: str,
) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None, "PowerShell 7 is required to exercise the CI script"
    pandoc_dir = tmp_path / "installed pandoc"
    pandoc_dir.mkdir()
    pandoc = pandoc_dir / "pandoc.cmd"
    pandoc.write_text(
        f'@echo off\nif not "%1"=="--version" exit /b 99\nexit /b {pandoc_exit_code}\n',
        encoding="utf-8",
    )
    attempts_file = tmp_path / "attempts.txt"
    github_path = tmp_path / "github-path.txt"
    script = tmp_path / "install.ps1"
    setup = rf"""
        $ErrorActionPreference = 'Stop'
        $global:attempts = 0
        $installCodes = @({", ".join(map(str, install_codes))})
        function choco {{
          $index = [Math]::Min($global:attempts, $installCodes.Count - 1)
          $global:attempts++
          $global:LASTEXITCODE = $installCodes[$index]
        }}
        function Get-Command {{
          param($Name, $CommandType, $ErrorAction)
          if ($global:attempts -ge {available_after}) {{
            Microsoft.PowerShell.Core\Get-Command @PSBoundParameters
          }}
        }}
        function Start-Sleep {{ param($Seconds) }}
        try {{
    """
    script.write_text(
        dedent(setup)
        + _workflow_script("Install pandoc on windows")
        + "\n} finally { $global:attempts | Set-Content $env:TEST_ATTEMPTS }\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script)],
        env=dict(
            os.environ,
            TEST_ATTEMPTS=str(attempts_file),
            GITHUB_PATH=str(github_path),
            PATH=f"{pandoc_dir}{os.pathsep}{os.environ['PATH']}",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert (result.returncode == 0) is success, output
    assert int(attempts_file.read_text(encoding="utf-8-sig")) == attempts
    if success:
        exported_paths = github_path.read_text(encoding="utf-8-sig").splitlines()
        assert exported_paths == [str(pandoc_dir)]
        next_step = subprocess.run(
            [os.environ["COMSPEC"], "/d", "/c", "pandoc --version"],
            env=dict(
                os.environ,
                PATH=os.pathsep.join([*exported_paths, os.environ["PATH"]]),
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        assert next_step.returncode == 0, next_step.stdout + next_step.stderr
    else:
        assert "failed after 5 attempts" in output
        assert message in output
        assert not github_path.exists()
