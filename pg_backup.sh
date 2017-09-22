fn="$(pwd)/pg_backups/finnsyll-$(date +%d%b%Y-%H:%M)"
pg_dump finnsyll > $fn