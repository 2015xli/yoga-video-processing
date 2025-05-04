Add-Type -AssemblyName Microsoft.VisualBasic
Add-Type -AssemblyName System.Windows.Forms

# Define paths
$pathMason   = "\\XF-NAS\video\001-video-to-process\yoga-training\Mason"
$pathLucy    = "\\XF-NAS\video\001-video-to-process\yoga-training\Lucy"
$deviceNames = @("XF S10", "XF's S20")

# Helper: prompt Yes/No with customizable default button
function Prompt-YesNo {
    param(
        [string]$message,
        [string]$title,
        [Microsoft.VisualBasic.MsgBoxStyle]$defaultButton
    )
    return [Microsoft.VisualBasic.Interaction]::MsgBox(
        $message,
        ([Microsoft.VisualBasic.MsgBoxStyle]::YesNo -bor $defaultButton),
        $title
    )
}

# Locate MTP device
$shellApp  = New-Object -ComObject Shell.Application
$devices   = $shellApp.Namespace(17).Items()
$mtpDevice = $devices | Where-Object { $deviceNames -contains $_.Name }
if (-not $mtpDevice) {
    [Microsoft.VisualBasic.Interaction]::MsgBox(
        "USB device not found. Plug in and try again.",
        "USB Script Error",
        [Microsoft.VisualBasic.MsgBoxStyle]::Critical
    )
    exit
}

# Navigate to Screen Recordings folder
$internal = $mtpDevice.GetFolder().Items() | Where-Object { $_.Name -eq 'Internal storage' }
$dcim     = $internal.GetFolder().Items() | Where-Object { $_.Name -eq 'DCIM' }
$screen   = $dcim.GetFolder().Items() | Where-Object { $_.Name -eq 'Screen recordings' }
if (-not $screen) {
    [Microsoft.VisualBasic.Interaction]::MsgBox(
        "Could not find 'Screen recordings' folder on device.",
        "USB Script Error",
        [Microsoft.VisualBasic.MsgBoxStyle]::Critical
    )
    exit
}

$videos = $screen.GetFolder().Items() | Where-Object { $_.Name -match '\.(mp4|mkv|avi)$' }
if (-not $videos) {
    Write-Host "No video files detected."
    exit
}

foreach ($file in $videos) {
    $fileName = $file.Name
    $ext      = [IO.Path]::GetExtension($fileName)

    # Date handling logic
    $sysDT    = [datetime]$file.ExtendedProperty("System.DateModified")
    $sysLocal = $sysDT.ToLocalTime()
    $useDT    = $sysLocal

    if ($fileName -match 'Screen_Recording_(\d{8})_(\d{6})_') {
        $fnDate = $matches[1]
        $fnTime = $matches[2]
        $fileDT = [datetime]::ParseExact("${fnDate}${fnTime}", 'yyyyMMddHHmmss', $null)
        
        if ($fileDT.Date -ne $sysLocal.Date) {
            $msg = "Filename date: $($fileDT.ToString('yyyy-MM-dd'))`r`n" +
                   "Modified date: $($sysLocal.ToString('yyyy-MM-dd'))`r`n" +
                   "Use filename date?"
            $ans = Prompt-YesNo $msg "Date Conflict" ([Microsoft.VisualBasic.MsgBoxStyle]::DefaultButton1)
            $useDT = if ($ans -eq 6) { $fileDT } else { $sysLocal }
        } else {
            $useDT = $fileDT
        }
    }

    # Target directory logic
    switch ($useDT.DayOfWeek) {
        'Tuesday'  { $defaultTarget = $pathMason }
        'Saturday' { $defaultTarget = $pathMason }
        default    { $defaultTarget = $pathLucy }
    }

    # Prompt configuration
    $defaultName = if ($defaultTarget -eq $pathMason) { 'Mason' } else { 'Lucy' }
    $otherName = if ($defaultName -eq 'Mason') { 'Lucy' } else { 'Mason' }
    
    $msgTgt = @"
Choose destination for:
"$fileName"

Yes = Move to $defaultName
No = Move to $otherName
"@

    # defaultBtn is always Yes (i.e., DefaultButton1), since Yes is for Move to $defaultName
    $defaultBtn = [Microsoft.VisualBasic.MsgBoxStyle]::DefaultButton1
    #$defaultBtn = if ($defaultName -eq 'Mason') {
    #    [Microsoft.VisualBasic.MsgBoxStyle]::DefaultButton1
    #} else {
    #    [Microsoft.VisualBasic.MsgBoxStyle]::DefaultButton2
    #}

    $ansTgt = Prompt-YesNo $msgTgt "Destination Selection" $defaultBtn
    $baseTarget = if ($ansTgt -eq 6) { $defaultTarget } else { (($pathLucy, $pathMason) -ne $defaultTarget)[0] }

    # File operations
    $subDir  = $useDT.ToString('yyyy-MM')
    $destDir = Join-Path $baseTarget $subDir
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }

    $newName = $useDT.ToString('yyyyMMdd_HHmmss') + $ext
    Write-Host "Processing: $fileName -> $destDir\$newName"
    
    try {
        $shellApp.Namespace($destDir).CopyHere($file, 16)
        do {
            Start-Sleep -Milliseconds 500
        } until (Test-Path (Join-Path $destDir $fileName))
        
        Rename-Item -LiteralPath (Join-Path $destDir $fileName) -NewName $newName
        $file.InvokeVerb("delete")
        Write-Host "Successfully moved: $newName"
    } catch {
        Write-Host "Error processing $fileName : $_"
    }
}

Write-Host "Operation completed successfully."