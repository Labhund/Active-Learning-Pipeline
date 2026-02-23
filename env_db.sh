#!/bin/bash
# ZINC20 HTVS Environment Configuration

# Define paths relative to the project root on the NVMe
export PROJ_ROOT="/data2/loo_lab/markus/analgesics"
export PGDATA="$PROJ_ROOT/data/postgres/data"
export PGHOST="$PROJ_ROOT/data/postgres/data" # Socket location
export PGLOG="$PROJ_ROOT/data/postgres/server.log"

# Shorthand Aliases for Workflow
alias db-start="pg_ctl -D $PGDATA -l $PGLOG start"
alias db-stop="pg_ctl -D $PGDATA stop"
alias db-status="pg_ctl -D $PGDATA status"
alias db-sql="psql -d analgesics"
alias db-schema="pg_dump -h \$PGHOST -s analgesics"

echo "HTVS Environment Loaded."
echo "DB Socket: $PGHOST"
