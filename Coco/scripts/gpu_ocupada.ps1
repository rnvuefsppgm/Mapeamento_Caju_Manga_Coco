# exit 0 = GPU ocupada pela manga (cadeia run_pos_treino_v1.sh ou qualquer processo MAPA_MANGA) ou por um 30_treino
# do coco lançado por ela; exit 1 = livre. Usado por run_treino_mapa.sh.
$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|bash).exe$' -and $_.CommandLine -and $_.CommandLine -notmatch 'Get-CimInstance|gpu_ocupada' -and $_.CommandLine -match 'run_pos_treino_v1|MAPA_MANGA|30_treino' }
if ($p) { exit 0 } else { exit 1 }
