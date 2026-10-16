$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$qaPath = Join-Path $projectRoot 'artifacts/presentation_notes_qa'
New-Item -ItemType Directory -Force -Path $qaPath | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $document = $word.Documents.Open((Join-Path $projectRoot 'DaisyRec Presentation Notes.docx'), $false, $true)
    $document.Repaginate()
    Write-Output ('Pages: ' + $document.ComputeStatistics(2))
    $document.ActiveWindow.View.Type = 3
    Add-Type -AssemblyName System.Drawing
    for ($pageIndex = 1; $pageIndex -le $document.ComputeStatistics(2); $pageIndex++) {
        $pageBytes = $document.ActiveWindow.Panes.Item(1).Pages.Item($pageIndex).EnhMetaFileBits
        $memoryStream = New-Object System.IO.MemoryStream(,$pageBytes)
        $metafile = [System.Drawing.Image]::FromStream($memoryStream)
        $bitmap = New-Object System.Drawing.Bitmap(1240,1754)
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
        $graphics.Clear([System.Drawing.Color]::White)
        $graphics.DrawImage($metafile,0,0,1240,1754)
        $bitmap.Save((Join-Path $qaPath "page-$pageIndex.png"),[System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose()
        $bitmap.Dispose()
        $metafile.Dispose()
        $memoryStream.Dispose()
        Write-Output "Rendered page $pageIndex"
    }
    $document.Close(0)
} finally {
    $word.Quit()
}


