<#
.SYNOPSIS
    Backup opencode session logs to git repo (opencodebackup).
    Копирует логи разговоров из ~/.local/share/opencode/log/ в git и пушит.
#>

$ErrorActionPreference = "Stop"
$OCODE_DIR = "$env:USERPROFILE\.local\share\opencode"
$BACKUP_DIR = "$env:USERPROFILE\Desktop\opencodebackup"
$TS = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

Write-Host "[$TS] Backup opencode sessions..."

# Init repo
if (-not (Test-Path -LiteralPath "$BACKUP_DIR\.git")) {
    New-Item -ItemType Directory -Path $BACKUP_DIR -Force | Out-Null
    Set-Location -LiteralPath $BACKUP_DIR
    git init
    git commit --allow-empty -m "init"
    Write-Host "Repo initialized at $BACKUP_DIR"
}

# Copy logs
$log_src = "$OCODE_DIR\log"
if (-not (Test-Path -LiteralPath $log_src)) {
    Write-Host "No log directory found at $log_src"
    exit 0
}

Remove-Item -LiteralPath "$BACKUP_DIR\log" -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item -LiteralPath $log_src -Destination $BACKUP_DIR -Recurse -Force
if (Test-Path -LiteralPath "$BACKUP_DIR\log") {
    $log_files = Get-ChildItem -LiteralPath "$BACKUP_DIR\log" -File
    $log_size = ($log_files | Measure-Object -Property Length -Sum).Sum
    $log_count = $log_files.Count
    Write-Host "Copied $log_count log files ($($log_size/1e6) MB)"
}

# Copy tool-output (последние выполнения команд)
$to_src = "$OCODE_DIR\tool-output"
if (Test-Path -LiteralPath $to_src) {
    Remove-Item -LiteralPath "$BACKUP_DIR\tool-output" -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item -LiteralPath $to_src -Destination $BACKUP_DIR -Recurse -Force
    $to_count = (Get-ChildItem -LiteralPath "$BACKUP_DIR\tool-output" -File).Count
    Write-Host "Copied $to_count tool-output files"
}

# Commit
Set-Location -LiteralPath $BACKUP_DIR
git add -A
$result = git commit -m "sessions: $TS" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Committed: $(git log --oneline -1)"
}
else {
    Write-Host "No changes"
}

# Push if OPENCODE_REMOTE set
$remote = $env:OPENCODE_REMOTE
if ($remote) {
    $existing = git remote get-url origin 2>$null
    if (-not $existing) { git remote add origin $remote }
    Write-Host "Pushing..."
    git push origin master 2>&1
    if ($LASTEXITCODE -eq 0) { Write-Host "Push OK" }
    else { Write-Host "Push failed" }
}
else {
    Write-Host "To push: `$env:OPENCODE_REMOTE = 'https://github.com/YOU/opencodebackup.git'"
}

Set-Location -LiteralPath "$(Split-Path -Parent $MyInvocation.MyCommand.Path)" -ErrorAction SilentlyContinue
Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Done."
