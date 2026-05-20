function convert-avitomp4 {
    param(
        [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
        [string[]]$InputFiles
    )

    $ffmpeg = "C:\Program Files\ffmpeg\ffmpeg-2026-05-18-git-b4d11dffbf-full_build\bin\ffmpeg.exe"
    $outputDir = "E:\videos_sleap_models"

    if (!(Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir | Out-Null
    }

    foreach ($InputFile in $InputFiles) {

        # Extract folder path info
        $subMatch = [regex]::Match($InputFile, "sub-(\d+)")
        $sesMatch = [regex]::Match($InputFile, "ses-(\d+)")

        $sub = $subMatch.Groups[1].Value
        $ses = $sesMatch.Groups[1].Value

        # Remove leading zeros (059 → 59)
        $sub = [int]$sub
        $ses = [int]$ses

        $suffix = "sub_${sub}_ses_${ses}"

        # Base filename (original video name without extension)
        $baseName = [System.IO.Path]::GetFileNameWithoutExtension($InputFile)

        $outputFile = Join-Path $outputDir "${baseName}_${suffix}.mp4"

        Write-Host "Converting: $InputFile -> $outputFile"

        & $ffmpeg `
            -y `
            -i $InputFile `
            -c:v h264_nvenc `
            -preset p7 `
            -rc vbr_hq `
            -cq 18 `
            -b:v 0 `
            -c:a aac `
            -b:a 192k `
            $outputFile
    }
}