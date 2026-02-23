#!/bin/bash
SESSION="HTVS_MONITOR"

# Create session and first pane (Main Ingestion Script)
tmux new-session -d -s $SESSION -n "Ingestion"
tmux send-keys -t $SESSION "source env_db.sh && python scripts/ligand_prep/ingest_zinc.py" C-m

# Split horizontally: Top Right for btop (Resource Monitor)
tmux split-window -h -t $SESSION:0
tmux send-keys -t $SESSION:0.1 "btop" C-m

# Split bottom right vertically: Bottom Right for DB Stats
tmux split-window -v -t $SESSION:0.1
tmux send-keys -t $SESSION:0.2 "watch -n 5 './scripts/utils/db_stats.sh'" C-m

# Re-focus the main process
tmux select-pane -t 0
tmux attach-session -t $SESSION
