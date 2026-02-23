-- Migration: add experiment_id column to al_batches and docking_scores.
-- Run once: psql -d analgesics -f scripts/utils/add_experiment_id.sql
--
-- Existing rows (round-0 MaxMin diversity init) correctly inherit
-- experiment_id = 'maxmin_init' via the DEFAULT clause.

ALTER TABLE al_batches     ADD COLUMN IF NOT EXISTS experiment_id TEXT NOT NULL DEFAULT 'maxmin_init';
ALTER TABLE docking_scores ADD COLUMN IF NOT EXISTS experiment_id TEXT NOT NULL DEFAULT 'maxmin_init';

CREATE INDEX IF NOT EXISTS idx_al_batches_exp     ON al_batches     (target, experiment_id, round);
CREATE INDEX IF NOT EXISTS idx_docking_scores_exp ON docking_scores (target, experiment_id, al_round);
