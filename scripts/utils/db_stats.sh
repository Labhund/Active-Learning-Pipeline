#!/bin/bash
# ZINC20 HTVS Ingestion Pulse

psql -d analgesics -c "
SELECT 
    (SELECT count(*) FROM public.compounds) as molecules,
    (SELECT count(*) FROM public.tranche_status WHERE status = 'completed') as tranches_done,
    (SELECT count(*) FROM public.tranche_status WHERE status = 'processing') as active_workers,
    pg_size_pretty(pg_total_relation_size('public.compounds')) as table_size
;"
