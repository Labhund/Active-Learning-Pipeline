-- AL pipeline database migration
-- Run once: psql -d analgesics -f scripts/utils/create_al_tables.sql

CREATE TABLE IF NOT EXISTS docking_scores (
    id          SERIAL PRIMARY KEY,
    compound_id INT     NOT NULL REFERENCES compounds(id),
    target      TEXT    NOT NULL,
    score       FLOAT,           -- NULL for failed docks; kcal/mol (negative = good)
    al_round    INT     NOT NULL,
    docked_at   TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ds_target_round ON docking_scores (target, al_round);
CREATE INDEX IF NOT EXISTS idx_ds_compound     ON docking_scores (compound_id, target);

CREATE TABLE IF NOT EXISTS al_batches (
    id          SERIAL PRIMARY KEY,
    round       INT     NOT NULL,
    compound_id INT     NOT NULL REFERENCES compounds(id),
    target      TEXT    NOT NULL,
    source      TEXT    NOT NULL   -- 'diversity_init' | 'thompson_sample'
);
CREATE INDEX IF NOT EXISTS idx_ab_round_target ON al_batches (round, target);

CREATE TABLE IF NOT EXISTS surrogate_predictions (
    round           INT   NOT NULL,
    compound_id     INT   NOT NULL REFERENCES compounds(id),
    target          TEXT  NOT NULL,
    predicted_score FLOAT NOT NULL,
    PRIMARY KEY (round, compound_id, target)
);
CREATE INDEX IF NOT EXISTS idx_sp_round_target ON surrogate_predictions (round, target);
