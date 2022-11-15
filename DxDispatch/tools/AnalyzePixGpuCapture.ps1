<#
.SYNOPSIS
Analyzes a PIX GPU capture recorded by dxdispatch.exe.
#>
param
(
    # Path to either a PIX GPU capture file (.wpix) file or event list (.csv) from a previously analyzed PIX capture file.
    [Parameter(Mandatory)][string]$InputPath,
    
    # Path to pixtool.exe. If not provided, the latest installed version is used.
    [string]$PixtoolPath
)

function PrettyDurationFormat($DurationNs)
{
    if ($DurationNs -ge 1000000)
    {
        return "{0:f2} ms" -f ($DurationNs / 1000000.0)
    }
    if ($DurationNs -ge 1000)
    {
        return "{0:f2} μs" -f ($DurationNs / 1000.0)
    }
    return "{0:f2} ns" -f $DurationNs
}

$InputFile = Get-ChildItem $InputPath -ErrorAction Ignore
if (!$InputFile)
{
    Write-Warning "Cannot find '$InputFile'"
    exit 1
}

if ($InputFile.Extension -eq '.wpix')
{
    if (!$PixtoolPath -and (Test-Path "$env:ProgramFiles\Microsoft PIX"))
    {
        $PixInstallDir = (Get-ChildItem "$env:ProgramFiles\Microsoft PIX" | Select-Object -Last 1).FullName
        $PixtoolPath = Join-Path $PixInstallDir "pixtool.exe"
    }
    
    if (!(Test-Path $PixtoolPath))
    {
        Write-Warning "Cannot not find pixtool.exe. Make sure PIX is installed, or set the path using -PixtoolPath."
        exit 1
    }
    
    # Pixtool expects a fully qualified path.
    $PixCaptureFile = (Resolve-Path $InputPath).Path
    
    $CounterCommandLine = 
        '--counters="Execution Start Time*"',
        '--counters="TOP to EOP Duration*"',
        '--counters="CS Invocations"'
    
    $PixCaptureEventsFileName = "events.csv"
    & $PixtoolPath open-capture $PixCaptureFile save-event-list $CounterCommandLine $PixCaptureEventsFileName
}
elseif ($InputFile.Extension -eq '.csv')
{
    $PixCaptureEventsFileName = $InputPath
}

$Events = Import-Csv $PixCaptureEventsFileName

$DmlOpDurationsNs = @{}
$DmlOpMetacommands = @{}
$ModelDurationNs = @{}

Write-Host ""

foreach ($Event in $Events)
{
    if (($Event.Name -eq 'Dispatch') -or ($Event.Name -eq 'ExecuteMetaCommand'))
    {
        # Get the top-level DML operator name
        $DmlOpName = "Other"
        $CurrentEvent = $Event
        while ($CurrentEvent.Parent -ge 0)
        {
            $CurrentEvent = $Events[$CurrentEvent.Parent]
            if ($CurrentEvent.Name.StartsWith('DML_OPERATOR'))
            {
                $DmlOpName = $CurrentEvent.Name -replace '(DML_OPERATOR_\w+).*','$1'
            }
            elseif ($CurrentEvent.Name.StartsWith('HLSL'))
            {
                $DmlOpName = $CurrentEvent.Name -replace 'HLSL: (.*)','$1'
            }
        }

        if ($DmlOpName -ne "Other")
        {
            $DmlOpDurationsNs[$DmlOpName] += [int]$Event.'TOP to EOP Duration (ns)'
        }

        if ($Event.Name -eq 'ExecuteMetaCommand')
        {
            $DmlOpMetacommands[$DmlOpName] = $True
        }
    }
    elseif ($Event.Name -match 'ONNX: ''(.*)''')
    {
        $ModelDurationNs[$Matches[1]] = [int]$Event.'TOP to EOP Duration (ns)'
    }
}

if ($ModelDurationNs.Count -gt 0)
{
    Write-Host "Model GPU Time:"
    foreach ($ModelName in $ModelDurationNs.Keys)
    {
        "  '{0}' : {1}" -f $ModelName, (PrettyDurationFormat $ModelDurationNs[$ModelName])
    }
    Write-Host ""
}

if ($DmlOpDurationsNs.Count -gt 0)
{
    $DmlOpsSortedByDuration = $DmlOpDurationsNs.Keys | Sort-Object { $DmlOpDurationsNs[$_] } -Descending
    $MaxNameLength = ($DmlOpsSortedByDuration | Measure-Object -Property Length -Maximum).Maximum

    $TotalOpTimeNs = 0
    foreach ($DmlOpName in $DmlOpsSortedByDuration)
    {
        $TotalOpTimeNs += $DmlOpDurationsNs[$DmlOpName]
    }

    Write-Host "Operator GPU Time (sum may exceed model time due to parallelism):"
    foreach ($DmlOpName in $DmlOpsSortedByDuration)
    {
        $DurationNs = $DmlOpDurationsNs[$DmlOpName]
        $DurationPct = $DmlOpDurationsNs[$DmlOpName] / $TotalOpTimeNs
        "  {0,-$MaxNameLength} : {1:f2} ({2:f1}%)" -f $DmlOpName, (PrettyDurationFormat $DurationNs), ($DurationPct * 100)
    }
    Write-Host ""
}

<#
Additional potential info to report:
- IDMLOperatorInitializer time
- Metacommands used
- Number of command lists
- Execution plan usage
- Memory stats
- Other PIX/GPU counters
#>