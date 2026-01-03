# Check Docker Desktop Status and Wait for Ready

Write-Host "=== Checking Docker Desktop Status ===" -ForegroundColor Cyan
Write-Host ""

$maxAttempts = 30
$attempt = 0

while ($attempt -lt $maxAttempts) {
    try {
        $result = docker ps 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Docker Desktop is running!" -ForegroundColor Green
            Write-Host ""
            Write-Host "You can now run:" -ForegroundColor Yellow
            Write-Host "  docker compose up -d" -ForegroundColor White
            exit 0
        }
    } catch {
        # Continue checking
    }
    
    $attempt++
    Write-Host "Waiting for Docker Desktop... ($attempt/$maxAttempts)" -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "❌ Docker Desktop did not start in time" -ForegroundColor Red
Write-Host ""
Write-Host "Please:" -ForegroundColor Yellow
Write-Host "1. Check if Docker Desktop is running (whale icon in system tray)" -ForegroundColor White
Write-Host "2. If not, start it manually from Start Menu" -ForegroundColor White
Write-Host "3. Wait for it to fully initialize" -ForegroundColor White
Write-Host "4. Then run this script again or try: docker ps" -ForegroundColor White
exit 1

