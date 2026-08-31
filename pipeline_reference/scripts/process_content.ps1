# Process all PDFs in the Content directory

$ContentDir = "Content"
$OutputBase = "data\output"

# Get all PDFs in the Content folder
$PdfFiles = Get-ChildItem -Path "$ContentDir\*.pdf"

Write-Host "=========================================================="
Write-Host "Starting batch processing of $($PdfFiles.Count) PDFs..."
Write-Host "Outputs will be saved in separated files in the $OutputBase directory."
Write-Host "=========================================================="

$i = 1
foreach ($Pdf in $PdfFiles) {
    Write-Host ""
    Write-Host "[$i/$($PdfFiles.Count)] Processing: $($Pdf.Name)"
    Write-Host "----------------------------------------------------------"
    
    # Run the docling_pipeline
    python -m extraction.docling_pipeline "$($Pdf.FullName)"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$i/$($PdfFiles.Count)] SUCCESS: $($Pdf.Name)" -ForegroundColor Green
    } else {
        Write-Host "[$i/$($PdfFiles.Count)] FAILED: $($Pdf.Name)" -ForegroundColor Red
    }
    
    $i++
}

Write-Host ""
Write-Host "=========================================================="
Write-Host "Batch processing complete!"
Write-Host "Check the $OutputBase directory for the separated output files."
Write-Host "=========================================================="
