{   # LOCAL ATTEMP

    # back up the metric-gold database
    fn="$(pwd)/pg_backups/finnsyll-$(date +%d%b%Y-%H:%M)"
    pg_dump finnsyll > $fn

    # delete backups that are more than 10 days old
    find "$(pwd)/pg_backups" -mtime +10 -exec rm {} \;

} || {  # SERVER ATTEMPT

    # back up the metric-gold database
    fn="~/finnsyll-dev/pg_backups/finnsyll-$(date +%d%b%Y-%H:%M)"
    pg_dump finnsyll > $fn

    # delete backups that are more than 10 days old
    find "~/finnsyll-dev/pg_backups" -mtime +10 -exec rm {} \;

}