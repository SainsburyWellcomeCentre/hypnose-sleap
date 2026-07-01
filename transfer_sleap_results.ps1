param(
    [string]$SourceRoot = "E:\derivatives",
    [string]$DestinationRoot = "Z:\hypnose\derivatives",
    # Match anywhere in name to catch prefixes like sub-XXX_ before the pattern
    [string[]]$Patterns = @("*sleap_tracking_video*.parquet", "*sleap_tracking_video*.csv", "*combined_sleap_tracking_timestamps*.parquet", "*combined_sleap_tracking_timestamps*.csv"),
    [switch]$DryRun,
    [switch]$ShowSkipped,
    [string[]]$Sub,
    [string[]]$Date
)

# Normalize trailing separators; use single-character trims
$normalizedSource = $SourceRoot.TrimEnd([char[]]"\\/")
$normalizedDest = $DestinationRoot.TrimEnd([char[]]"\\/")

if (-not (Test-Path -LiteralPath $normalizedSource)) {
    Write-Error "Source root not found: $normalizedSource"
    exit 1
}

$copied = 0
$skipped = 0

# Build sub filter set (integers) if provided
$subSet = $null
if ($Sub -and $Sub.Count -gt 0) {
    $subSet = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($s in $Sub) {
        $parsed = 0
        if ([int]::TryParse($s, [ref]$parsed)) {
            $subSet.Add($parsed) | Out-Null
        }
    }
    if ($subSet.Count -eq 0) { $subSet = $null }
}

# Build date filters (list of single ints and ranges)
$dateSingles = $null
$dateRanges = $null
if ($Date -and $Date.Count -gt 0) {
    $dateSingles = [System.Collections.Generic.HashSet[int]]::new()
    $dateRanges = New-Object System.Collections.Generic.List[object]
    foreach ($d in $Date) {
        if ($d -match "^([0-9]{8})-([0-9]{8})$") {
            $start = [int]$Matches[1]
            $end = [int]$Matches[2]
            if ($start -le $end) { $dateRanges.Add(@($start, $end)) }
            elseif ($start -gt $end) { $dateRanges.Add(@($end, $start)) }
        }
        elseif ($d -match "^[0-9]{8}$") {
            $dateSingles.Add([int]$d) | Out-Null
        }
    }
    if ($dateSingles.Count -eq 0 -and $dateRanges.Count -eq 0) {
        $dateSingles = $null
        $dateRanges = $null
    }
}

function Test-DateAllowed {
    param(
        [int]$DateValue,
        $Singles,
        $Ranges
    )

    if (-not $Singles -and -not $Ranges) { return $true }
    if ($Singles -and $Singles.Contains($DateValue)) { return $true }
    if ($Ranges) {
        foreach ($r in $Ranges) {
            if ($DateValue -ge $r[0] -and $DateValue -le $r[1]) { return $true }
        }
    }
    return $false
}

Get-ChildItem -LiteralPath $normalizedSource -Recurse -File | ForEach-Object {
    $matches = $false
    foreach ($pattern in $Patterns) {
        if ($_.Name -like $pattern) { $matches = $true; break }
    }

    if (-not $matches) {
        $skipped++
        if ($ShowSkipped) {
            $relSkip = $_.FullName.Substring($normalizedSource.Length).TrimStart([char[]]"\\/")
            Write-Host "Skipped (non-matching): $relSkip"
        }
        return
    }

    # Extract sub and date identifiers from path parts for optional filtering
    $relativePath = $_.FullName.Substring($normalizedSource.Length).TrimStart([char[]]"\\/")
    $parts = $relativePath -split "[\\/]"
    $subFolder = if ($parts.Count -ge 1) { $parts[0] } else { $null }
    $sesFolder = if ($parts.Count -ge 2) { $parts[1] } else { $null }

    $subOk = $true
    if ($subSet -and $subFolder -match "^sub-([0-9]{1,3})") {
        $subNum = [int]$Matches[1]
        $subOk = $subSet.Contains($subNum)
    }
    elseif ($subSet) {
        $subOk = $false
    }

    $dateOk = $true
    if (($dateSingles -or $dateRanges) -and $sesFolder -match "date-([0-9]{8})") {
        $dateVal = [int]$Matches[1]
        $dateOk = Test-DateAllowed -DateValue $dateVal -Singles $dateSingles -Ranges $dateRanges
    }
    elseif ($dateSingles -or $dateRanges) {
        $dateOk = $false
    }

    if (-not ($subOk -and $dateOk)) {
        $skipped++
        if ($ShowSkipped) {
            $reason = @()
            if (-not $subOk) { $reason += "sub filter" }
            if (-not $dateOk) { $reason += "date filter" }
            Write-Host "Skipped (" + ($reason -join ", ") + "): $relativePath"
        }
        return
    }

    $destPath = Join-Path $normalizedDest $relativePath
    $destDir = Split-Path -Path $destPath -Parent

    if (-not (Test-Path -LiteralPath $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    if ($DryRun) {
        Write-Host "Would copy: $($_.FullName) -> $destPath"
    }
    else {
        Copy-Item -LiteralPath $_.FullName -Destination $destPath -Force
        Write-Host "Copied: $($_.FullName) -> $destPath"
    }

    $copied++
}

Write-Host "Done. Copied: $copied. Skipped (non-matching): $skipped."